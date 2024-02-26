import mujoco as mj
import mujoco.viewer as mjv
from envs.nightmare_v3_config import NightmareV3Config
from envs.helpers import class_to_dict
import numpy as np

# print(mj.mj_step_threaded)
import torch

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


class NightmareV3Env():
    def __init__(self, cfg: NightmareV3Config):
        self.cfg = cfg

        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_obs
        self.num_privileged_obs = self.num_obs
        self.num_actions = self.cfg.env.num_actions

        self.model = mj.MjModel.from_xml_path(self.cfg.env.model_path)
        self.data = [mj.MjData(self.model) for _ in range(self.num_envs)]
        self.prev_data = [mj.MjData(self.model) for _ in range(self.num_envs)]

        self.num_dof = self.model.nv - 6
        self.num_geoms = self.model.ngeom
        print("Number of DoF: ", self.num_dof)

        self.gravity_vec = torch.tensor([0., 0., -9.81], dtype=torch.float, device=self.cfg.device, requires_grad=False)

        # useful buffers
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.contact_forces = torch.zeros(self.num_envs, self.num_geoms, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.base_heights = torch.zeros(self.num_envs, dtype=torch.float, device=self.cfg.device, requires_grad=False)

        self.termination_contact_geom_indices = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name) for name in self.cfg.env.feet_names]
        self.feet_geom_indices = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name) for name in self.cfg.env.feet_names]
        self.floor_geom_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, self.cfg.env.floor_name)
        self.body_geom_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, self.cfg.env.body_name)

        self.body_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.cfg.env.body_name)

        # check no index is -1
        assert all([index != -1 for index in self.termination_contact_geom_indices])
        assert all([index != -1 for index in self.feet_geom_indices])
        assert self.floor_geom_index != -1
        assert self.body_geom_index != -1

        if self.cfg.viewer.render:
            self.viewer = mjv.launch_passive(self.model, self.data[0])
        
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = self.cfg.commands.ranges
        self.obs_scales = self.cfg.normalization.obs_scales

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.feet_air_time = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, 6, dtype=torch.bool, device=self.cfg.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.cfg.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.cfg.device, requires_grad=False,) # TODO change this
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.cfg.device, requires_grad=False)

        self.dt = self.cfg.env.dt
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.default_dof_pos = torch.tensor(self.cfg.control.default_pos, dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.default_dof_pos_np = self.default_dof_pos.numpy()

        self.extras = {}

        self.common_step_counter = 0

        self.noise_scale_vec = torch.zeros_like(self.obs_buf[0])
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
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.cfg.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.prev_actions = self.actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.cfg.device)
        # step physics and render each frame
        self.render()

        prev_dof_vel = torch.tensor(np.array([self.prev_data[env_id].qvel[-18:] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)

        temp = np.array(self.actions) * self.cfg.control.action_scale - self.default_dof_pos_np
        self.prev_data = self.data.copy()

        ### single thread
        # for env_id in range(self.num_envs):
        #     self.data[env_id].ctrl = temp[env_id]
        #     mj.mj_step(self.model, self.data[env_id], self.cfg.control.decimation)

        ### multi thread
        for env_id in range(self.num_envs):
            self.data[env_id].ctrl = temp[env_id]
        mj.mj_step_multithreaded(self.model, self.data, self.cfg.control.decimation, 12)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # update useful buffers
        self.base_quat = torch.tensor(np.array([self.data[env_id].qpos[3:7] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.base_lin_vel = torch.tensor(np.array([self.data[env_id].cvel[self.body_index][3:6] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        
        self.base_lin_vel = quaternion_apply(self.base_quat, self.base_lin_vel)

        self.base_ang_vel = torch.tensor(np.array([self.data[env_id].cvel[self.body_index][:3] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.projected_gravity = self.gravity_vec.repeat(self.num_envs, 1)

        self.projected_gravity = quaternion_apply(self.base_quat, self.projected_gravity)

        self.dof_pos = torch.tensor(np.array([self.data[env_id].qpos[-18:] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.dof_vel = torch.tensor(np.array([self.data[env_id].qvel[-18:] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        self.torques = torch.tensor(np.array([self.data[env_id].qfrc_applied[-18:] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)
        
        self.dof_acc = (self.dof_vel - prev_dof_vel) / self.dt

        self.base_heights = torch.tensor(np.array([self.data[env_id].xipos[1][2] for env_id in range(self.num_envs)]), dtype=torch.float, device=self.cfg.device, requires_grad=False)

        # update contact forces
        # self.contact_forces = torch.zeros_like(self.contact_forces)
        # for env_id in range(self.num_envs):
        #     for contact in self.data[env_id].contact:
        #         self.contact_forces[env_id, contact.geom1] = self.data[env_id].efc_force[contact.efc_address]
        #         self.contact_forces[env_id, contact.geom2] = self.data[env_id].efc_force[contact.efc_address]

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # check for termination
        self.reset_buf = torch.any(self.contact_forces[:, self.termination_contact_geom_indices] > self.cfg.env.termination_contact_force, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        # compute rewards
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        # reset some environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # compute observations
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.cfg.device) * 2 * self.command_ranges.max_lin_vel_x - self.command_ranges.max_lin_vel_x
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.cfg.device) * 2 * self.command_ranges.max_lin_vel_y - self.command_ranges.max_lin_vel_y
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.cfg.device) * 2 * self.command_ranges.max_ang_vel - self.command_ranges.max_ang_vel

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

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
            self.prev_data[env_id].qpos = self.model.qpos0
            self.prev_data[env_id].qvel = 0.

        self._resample_commands(env_ids)

        # reset buffers
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def render(self):
        """ Render the environment"""
        if self.cfg.viewer.render:
            self.viewer.sync()

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.cfg.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.cfg.device, requires_grad=False))
        return obs, privileged_obs

    # rewards
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_heights - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.prev_actions - self.actions), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_geom_indices] > 0.004
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.01)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_geom_indices], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
