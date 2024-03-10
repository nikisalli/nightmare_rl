import mujoco as mj
import mujoco.viewer as mjv
from nikengine.engine import EngineNode, set_time_s, config
from pynput.keyboard import Key, Listener, Controller, KeyCode
import time
import pickle
import numpy as np
import torch
from envs.nightmare_v3_config import NightmareV3Config, NightmareV3ConfigPPO
from envs.helpers import class_to_dict, get_load_path
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent

# Dictionary to store the state of the keys
lin = -0.5
ang = 1.5

lin_speed = 0.0
ang_speed = 0.0

prev_time = 0

decimation = 4
action_rate = 0.08
prev_actions = np.zeros(18)

def vec_to_rot_matrix(vec):
    """Convert a 3D vector into a full three-dimensional rotation matrix. that rotates the z-axis to the vector."""
    vec = vec / np.linalg.norm(vec)
    cross = np.cross([0, 0, 1], vec)
    dot = np.dot([0, 0, 1], vec)
    cross_matrix = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + cross_matrix + np.matmul(cross_matrix, cross_matrix) * ((1 - dot) / (np.linalg.norm(cross) ** 2))

# Update the state of the keys when they are pressed or released
def on_key_change(key, state):
    global lin_speed, ang_speed
    if key == KeyCode.from_char('è'): lin_speed = lin * state
    elif key == KeyCode.from_char('à'): lin_speed = -lin * state
    elif key == KeyCode.from_char('ò'): ang_speed = ang * state
    elif key == KeyCode.from_char('ù'): ang_speed = -ang * state
    else: return

# Create the listener for the keyboard events
listener = Listener(on_press=lambda key: on_key_change(key, 1), 
                    on_release=lambda key: on_key_change(key, 0))
listener.start()

# model = mj.load_model_from_path("models/nightmare_v3/mjmodel.xml")
model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")

model.opt.timestep = 0.005
data = mj.MjData(model)

print("time step: ", model.opt.timestep)

accum = 0
prev = time.time()

qposs = []

# load model
cfg = NightmareV3Config()
train_cfg = NightmareV3ConfigPPO()
train_cfg_dict = class_to_dict(train_cfg)
actor_critic_class = eval(train_cfg_dict["runner"]["policy_class_name"])
nn = actor_critic_class( cfg.env.num_obs,
                                                cfg.env.num_obs,
                                                        cfg.env.num_actions,
                                                        **train_cfg_dict["policy"]).to('cpu')

model_path = "checkpoints/2024-03-10 02:54:19.368795/model_2000.pt"
nn.load_state_dict(torch.load(model_path)['model_state_dict'])
nn.eval()

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()

    prev_base_pos = np.zeros(3)
    default_pos = cfg.control.default_pos
    prev_actions = default_pos

    while viewer.is_running():
        set_time_s(data.time)

        base_quat = data.qpos[3:7]
        neg_base_quat = np.zeros_like(base_quat)
        mj.mju_negQuat(neg_base_quat, base_quat)
        gravity = np.array([0, 0, -9.81])
        projected_gravity = np.zeros_like(gravity)
        mj.mju_rotVecQuat(projected_gravity, gravity, neg_base_quat)
        # print(np.sum(np.square(projected_gravity[:2])))

        base_lin_vel = data.cvel[1][3:6]
        mj.mju_rotVecQuat(base_lin_vel, data.cvel[1][3:6], neg_base_quat)
        # print(np.linalg.norm(base_lin_vel))
        base_ang_vel = data.cvel[1][:3]
        mj.mju_rotVecQuat(base_ang_vel, data.cvel[1][:3], neg_base_quat)

        base_pos = data.qpos[:3]
        base_lin_vel2 = (base_pos - prev_base_pos) / (model.opt.timestep * decimation)
        mj.mju_rotVecQuat(base_lin_vel2, base_lin_vel2, neg_base_quat)
        prev_base_pos = base_pos.copy()

        dof_pos = data.qpos[-18:]
        dof_vel = data.qvel[-18:]

        command = np.array([lin_speed, 0, ang_speed])

        obs = np.concatenate((
            base_lin_vel * cfg.normalization.obs_scales.lin_vel,
            base_ang_vel * cfg.normalization.obs_scales.ang_vel,
            projected_gravity,
            command * np.array([cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.ang_vel]),
            (dof_pos - default_pos) * cfg.normalization.obs_scales.dof_pos,
            dof_vel * cfg.normalization.obs_scales.dof_vel,
            prev_actions
        ), axis=-1)

        # clip obs
        clip_obs = cfg.normalization.clip_observations
        obs = np.clip(obs, -clip_obs, clip_obs)

        print(obs)
        print(default_pos)

        actions = nn.act(torch.tensor(obs, dtype=torch.float32))

        clip_actions = cfg.normalization.clip_actions
        all_actions = np.array(actions.cpu().numpy()) * cfg.control.action_scale
        actions = np.array(np.clip(all_actions[:18], -clip_actions, clip_actions))[:18]
        prev_actions = actions.copy()
        actions = actions - default_pos

        data.ctrl[:] = (actions - dof_pos) * cfg.control.p_gain

        mj.mj_step(model, data, decimation)

        coxa_contact_forces = data.sensordata[:6].copy()
        femur_contact_forces = data.sensordata[6:12].copy()
        tibia_contact_forces = data.sensordata[12:18].copy()
        feet_contact_forces = data.sensordata[18:24].copy()

        tibia_contact_forces *= (feet_contact_forces == 0)

        viewer.sync()

        arrow_command = np.array([command[0], -command[2], 0])
        mj.mju_rotVecQuat(arrow_command, arrow_command, data.qpos[3:7])

        viewer.user_scn.ngeom = 0
        mj.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mj.mjtGeom.mjGEOM_ARROW, 
            size=np.array([0.02, 0.02, 1]),
            rgba=np.array([255, 255, 255, 255]), 
            pos=np.array([data.qpos[0], data.qpos[1], data.qpos[2] + 0.5]),
            # mat=vec_to_rot_matrix(projected_gravity).flatten()
            mat=vec_to_rot_matrix(arrow_command).flatten()
        )
        viewer.user_scn.ngeom = 1

        viewer.cam.lookat[0] = data.qpos[0]
        viewer.cam.lookat[1] = data.qpos[1]
        viewer.cam.lookat[2] = data.qpos[2]

        # wait for 0.002 seconds
        while time.time() - prev_time < model.opt.timestep * decimation:
            pass
        prev_time = time.time()

        accum += 1
        if accum > 1000:
            now = time.time()
            print(accum / (now - prev))
            accum = 0
            prev = now