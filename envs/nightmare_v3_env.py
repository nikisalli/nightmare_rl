import mujoco as mj
import mujoco.viewer as mjv
from envs.nightmare_v3_config import NightmareV3Config
from envs.helpers import class_to_dict
import numpy as np
import time
import threading
import pickle
import os

# print(mj.mj_step_threaded)
import torch

class NightmareV3Env():
    def __init__(self, cfg: NightmareV3Config, log_dir="/tmp/nightmare_v3/logs", num_threads=1):
        self.cfg = cfg
        self.log_dir = log_dir
        self.thread_num = num_threads

        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_obs
        self.num_privileged_obs = self.num_obs
        self.num_actions = self.cfg.env.num_actions

        self.model = mj.MjModel.from_xml_path(self.cfg.env.model_path)
        self.data = [mj.MjData(self.model) for _ in range(self.num_envs)]

        self.num_dof = self.model.nv - 6
        self.num_geoms = self.model.ngeom
        print("Number of DoF: ", self.num_dof)

        self.gravity_vec = np.array([0., 0., -9.81])

        # self.termination_contact_geom_indices = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name) for name in self.cfg.env.termination_names]
        # self.floor_geom_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, self.cfg.env.floor_name)
        # self.body_geom_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, self.cfg.env.body_name)
        self.body_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.cfg.env.body_name)

        # print("Termination contact geom indices: ", self.termination_contact_geom_indices)
        # print("Floor geom index: ", self.floor_geom_index)
        print("Body geom index: ", self.body_index)

        # check no index is -1
        # assert all([index != -1 for index in self.termination_contact_geom_indices])
        # assert self.floor_geom_index != -1
        # assert self.body_geom_index != -1

        # useful buffers
        self.base_quat = np.zeros((self.num_envs, 4))
        self.base_lin_vel = np.zeros((self.num_envs, 3))
        self.base_ang_vel = np.zeros((self.num_envs, 3))
        self.projected_gravity = np.zeros((self.num_envs, 3))
        self.dof_pos = np.zeros((self.num_envs, self.num_dof))
        self.dof_vel = np.zeros((self.num_envs, self.num_dof))
        self.contact_forces = np.zeros((self.num_envs, len(self.cfg.env.feet_names)))
        self.torques = np.zeros((self.num_envs, self.num_dof))
        self.prev_actions = np.zeros((self.num_envs, self.num_actions))
        self.base_heights = np.zeros(self.num_envs)

        if self.cfg.viewer.render:
            self.viewer = mjv.launch_passive(self.model, self.data[0])
        
        # save states to render offline
        self.recorded_states = []

        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = self.cfg.commands.ranges
        self.obs_scales = self.cfg.normalization.obs_scales

        self.obs_buf = np.zeros((self.num_envs, self.num_obs))
        self.privileged_obs_buf = None
        self.rew_buf = np.zeros(self.num_envs)
        self.reset_buf = np.ones(self.num_envs, dtype=np.int64)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64)
        self.time_out_buf = np.zeros(self.num_envs, dtype=bool)
        self.feet_air_time = np.zeros((self.num_envs, 6))
        self.leg_contact_force_difference = np.zeros((self.num_envs, 6))
        self.last_contacts = np.zeros((self.num_envs, 6), dtype=bool)
        self.commands = np.zeros((self.num_envs, 3))
        self.commands_scale = np.array([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel])
        self.actions = np.zeros((self.num_envs, self.num_actions))
        self.limited_actions = np.zeros((self.num_envs, self.num_actions))

        self.dt = self.cfg.env.dt
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.default_dof_pos = np.array(self.cfg.control.default_pos, dtype=np.float64)

        self.extras = {}

        self.common_step_counter = 0

        self.noise_scale_vec = np.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        self.noise_scale_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        self.noise_scale_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        self.noise_scale_vec[6:9] = noise_scales.gravity * noise_level
        self.noise_scale_vec[9:12] = 0. # commands
        self.noise_scale_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        self.noise_scale_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        self.noise_scale_vec[36:48] = 0. # previous actions

        # episode sums
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: np.zeros(self.num_envs) for name in self.reward_scales.keys()}

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (np.array): Array of shape (num_envs, num_actions_per_env)
        """

        time1 = time.time()
        clip_actions = self.cfg.normalization.clip_actions
        self.prev_actions = self.actions
        self.actions = np.clip(actions.cpu().numpy(), -clip_actions, clip_actions)

        time2 = time.time()
        # step physics and render each frame
        self.render()

        time3 = time.time()

        prev_dof_vel = self.dof_vel.copy()

        actions = np.array(self.actions) * self.cfg.control.action_scale - self.default_dof_pos
        self.limited_actions += np.clip(actions - self.limited_actions, -self.cfg.control.action_rate_limit, self.cfg.control.action_rate_limit)
        velocity_command = (self.limited_actions - self.dof_pos) * self.cfg.control.p_gain #  - self.cfg.control.d_gain * self.dof_vel

        time4 = time.time()

        ### multi thread
        for env_id in range(self.num_envs):
            self.data[env_id].ctrl = velocity_command[env_id]

        time5 = time.time()

        # python threading
        batch_size = self.num_envs // self.thread_num
        threads = []

        def step_thread(start, end):
            for i in range(start, end):
                mj.mj_step(self.model, self.data[i], self.cfg.control.decimation)

        for i in range(self.thread_num):
            start = i * batch_size
            end = (i + 1) * batch_size if i != self.thread_num - 1 else self.num_envs
            t = threading.Thread(target=step_thread, args=(start, end))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

        time6 = time.time()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        time7 = time.time()

        # update useful buffers
        for env_id in range(self.num_envs): self.base_quat[env_id] = self.data[env_id].qpos[3:7].copy()
        time8 = time.time()
        for env_id in range(self.num_envs): mj.mju_rotVecQuat(self.base_lin_vel[env_id], self.data[env_id].cvel[self.body_index][3:6], self.base_quat[env_id])
        time9 = time.time()
        time10 = time.time()
        time11 = time.time()
        time12 = time.time()
        for env_id in range(self.num_envs): self.base_ang_vel[env_id] = self.data[env_id].cvel[self.body_index][:3].copy()
        time13 = time.time()
        for env_id in range(self.num_envs): mj.mju_rotVecQuat(self.projected_gravity[env_id], self.gravity_vec, self.base_quat[env_id])
        time14 = time.time()
        for env_id in range(self.num_envs): self.dof_pos[env_id] = self.data[env_id].qpos[-18:].copy()
        time15 = time.time()
        for env_id in range(self.num_envs): self.dof_vel[env_id] = self.data[env_id].qvel[-18:].copy()
        time16 = time.time()
        for env_id in range(self.num_envs): self.torques[env_id] = self.data[env_id].qfrc_applied[-18:].copy()
        
        time17 = time.time()
        self.dof_acc = (self.dof_vel - prev_dof_vel) / self.dt

        for env_id in range(self.num_envs): self.base_heights[env_id] = self.data[env_id].xipos[1][2].copy()

        time18 = time.time()
        # update contact forces
        for env_id in range(self.num_envs): self.contact_forces[env_id] = self.data[env_id].sensordata[:6].copy()

        time19 = time.time()
        # env_ids = np.nonzero(self.episode_length_buf_np % int(self.cfg.commands.resampling_time / self.dt) == 0)[0]
        env_ids = np.array(np.nonzero(self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt / self.cfg.control.decimation) == 0).flatten())
        self._resample_commands(env_ids)

        time20 = time.time()

        # check for termination
        self.reset_buf = np.zeros_like(self.reset_buf)
        # self.reset_buf = np.any(self.contact_forces[:, self.termination_contact_geom_indices] > self.cfg.env.termination_contact_force, axis=1)
        # check max episode length termination
        # self.time_out_buf = self.episode_length_buf_np > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf = np.array(self.episode_length_buf) > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # check feet contact forces too high
        self.reset_buf |= self.contact_forces[:, :6].max(axis=1) > self.cfg.env.termination_contact_force
        # check robot not upside down
        max_angle = 60 * np.pi / 180 # 60 degrees
        vec_pointing_down = np.array([0, 0, -1])
        # np.arccos(np.dot(vec_pointing_down, projgrav) / (np.linalg.norm(vec_pointing_down) * np.linalg.norm(new_vec)))
        self.reset_buf |= np.arccos(np.dot(self.projected_gravity, vec_pointing_down) / (np.linalg.norm(self.projected_gravity * np.linalg.norm(vec_pointing_down), axis=1))) > max_angle
        # reset some environments
        env_ids = np.nonzero(self.reset_buf.flatten())[0]

        # record states
        if self.cfg.viewer.record_states:
            # check if it's time to save states
            if self.reset_buf[0] == True:
                # make log dir if not present
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                # save states in file like unixtimestamp.pkl
                with open(f"{self.log_dir}/{int(time.time())}.pkl", "wb") as f:
                    pickle.dump(self.recorded_states, f)
            self.recorded_states.append((self.data[0].time, self.data[0].qpos.copy(), self.data[0].qvel.copy(), self.data[0].act.copy()))

        self.reset_idx(env_ids)

        time21 = time.time()

        # compute rewards
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = np.array(self.reward_functions[i]()) * self.reward_scales[name]
            # print(f"Reward {name}: {rew[0]}")
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = np.array(self._reward_termination()) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        time22 = time.time()

        time23 = time.time()

        # compute observations
        self.obs_buf = np.concatenate((self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions), axis=-1)
        
        time24 = time.time()

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * np.random.rand(*self.obs_buf.shape) - 1) * self.noise_scale_vec

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = np.clip(self.obs_buf, -clip_obs, clip_obs)

        time25 = time.time()

        # print(f"Time2 - Time1: {(time2 - time1) * 1000} ms")
        # print(f"Time3 - Time2: {(time3 - time2) * 1000} ms")
        # print(f"Time4 - Time3: {(time4 - time3) * 1000} ms")
        # print(f"Time5 - Time4: {(time5 - time4) * 1000} ms")
        # print(f"Time6 - Time5: {(time6 - time5) * 1000} ms")
        # print(f"Time7 - Time6: {(time7 - time6) * 1000} ms")
        # print(f"Time8 - Time7: {(time8 - time7) * 1000} ms")
        # print(f"Time9 - Time8: {(time9 - time8) * 1000} ms")
        # # print(f"Time10 - Time9: {(time10 - time9) * 1000} ms")
        # # print(f"Time11 - Time10: {(time11 - time10) * 1000} ms")
        # # print(f"Time12 - Time11: {(time12 - time11) * 1000} ms")
        # print(f"Time13 - Time12: {(time13 - time12) * 1000} ms")
        # print(f"Time14 - Time13: {(time14 - time13) * 1000} ms")
        # print(f"Time15 - Time14: {(time15 - time14) * 1000} ms")
        # print(f"Time16 - Time15: {(time16 - time15) * 1000} ms")
        # print(f"Time17 - Time16: {(time17 - time16) * 1000} ms")
        # print(f"Time18 - Time17: {(time18 - time17) * 1000} ms")
        # print(f"Time19 - Time18: {(time19 - time18) * 1000} ms")
        # print(f"Time20 - Time19: {(time20 - time19) * 1000} ms")
        # print(f"Time21 - Time20: {(time21 - time20) * 1000} ms")
        # print(f"Time22 - Time21: {(time22 - time21) * 1000} ms")
        # print(f"Time23 - Time22: {(time23 - time22) * 1000} ms")
        # print(f"Time24 - Time23: {(time24 - time23) * 1000} ms")
        # print(f"Time25 - Time24: {(time25 - time24) * 1000} ms")
    
        # return tensors only
        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        return torch.tensor(self.obs_buf, dtype=torch.float32), None, torch.tensor(self.rew_buf, dtype=torch.float32), torch.tensor(self.reset_buf, dtype=torch.float32), self.extras

    def get_observations(self):
        # return self.obs_buf
        return torch.tensor(self.obs_buf, dtype=torch.float32)
    
    def get_privileged_observations(self):
        # return self.privileged_obs_buf
        return None
    
    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = np.random.rand(len(env_ids)) * 2 * self.command_ranges.max_lin_vel_x - self.command_ranges.max_lin_vel_x
        self.commands[env_ids, 1] = np.random.rand(len(env_ids)) * 2 * self.command_ranges.max_lin_vel_y - self.command_ranges.max_lin_vel_y
        self.commands[env_ids, 2] = np.random.rand(len(env_ids)) * 2 * self.command_ranges.max_ang_vel - self.command_ranges.max_ang_vel

        # set small commands to zero
        self.commands[env_ids, :2] *= (np.linalg.norm(self.commands[env_ids, :2], axis=1) > 0.2)[:, np.newaxis]

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        for env_id in env_ids:
            self.data[env_id].qpos = self.model.qpos0
            self.data[env_id].qvel = 0

        self._resample_commands(env_ids)

        # reset buffers
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # self.extras["episode"]['rew_' + key] = np.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.tensor(np.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s, dtype=torch.float32)
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            # self.extras["time_outs"] = self.time_out_buf
            self.extras["time_outs"] = torch.tensor(self.time_out_buf, dtype=torch.float32)

    def draw_arrow(self, pos, quat, size, rgba=(255, 255, 255, 255)):
        mat = np.zeros((9,), dtype=np.float64)
        mj.mju_quat2Mat(mat, quat)
        mj.mjv_initGeom(
            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW, 
            size=size, 
            rgba=rgba, 
            pos=pos, 
            mat=mat
        )
        self.viewer.user_scn.ngeom += 1

    def render(self):
        """ Render the environment"""
        if self.cfg.viewer.render:
            self.viewer.sync()
        # self.viewer.user_scn.ngeom = 0

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(np.arange(self.num_envs))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros((self.num_envs, self.num_actions)))
        return obs, privileged_obs

    # rewards
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.base_ang_vel[:, :2]), axis=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.projected_gravity[:, :2]), axis=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return np.square(self.base_heights - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques), axis=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return np.sum(np.square(self.dof_vel), axis=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square(self.dof_acc), axis=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.prev_actions - self.actions), axis=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), axis=1)
        return np.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = np.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return np.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact_filt = np.logical_or(self.contact_forces > 0.5, self.last_contacts)
        self.last_contacts = self.contact_forces
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = np.sum((self.feet_air_time - 0.5) * first_contact, axis=1) # reward only on first contact with the ground
        rew_airTime *= np.linalg.norm(self.commands[:, :2], axis=1) > 0.01 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.dof_pos - self.default_dof_pos), axis=1) * (np.linalg.norm(self.commands[:, :2], axis=1) < 0.01)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return np.sum(((self.contact_forces - self.cfg.rewards.max_contact_force) * (self.contact_forces > self.cfg.rewards.max_contact_force))**2, axis=1)
    
    def _reward_default_position(self):
        # penalize distance from default position
        return np.sum(np.square(self.dof_pos - self.default_dof_pos), axis=1)

    def _reward_contact_forces_difference(self):
        # penalize difference in contact forces between legs
        diff = self.contact_forces - np.mean(self.contact_forces, axis=1)[:, np.newaxis]
        self.leg_contact_force_difference += diff
        return np.sum(np.square(self.leg_contact_force_difference), axis=1)
