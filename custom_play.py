import mujoco as mj
import mujoco.viewer as mjv
from nikengine.engine import EngineNode, set_time_s, config
from pynput.keyboard import Key, Listener, Controller, KeyCode
import time
import pickle
import numpy as np

# Dictionary to store the state of the keys
lin = 0.05
ang = 0.2

lin_speed = 0.0
ang_speed = 0.0

prev_time = 0

decimation = 2
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

model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")
# model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel_mjx.xml")

data = mj.MjData(model)

engine = EngineNode()
engine.update(0.0, 0.0, 'idle')
config.ENGINE_FPS = 1.0 / model.opt.timestep / decimation
config.STAND_HEIGHT = 0.2

print("time step: ", model.opt.timestep)

accum = 0
prev = time.time()

qposs = []

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()

    prev_base_pos = np.zeros(3)

    while viewer.is_running():
        set_time_s(data.time)
        # print("time: ", data.time)
        actions = engine.update(lin_speed, ang_speed, 'awake', 'walk')
        # actions = np.random.uniform(-0.5, 0.5, 18)
        # limit action_rate
        prev_actions += np.clip(actions - prev_actions, -action_rate, action_rate)
        kp = 12
        data.ctrl[:] = (prev_actions - data.qpos[-18:]) * kp

        mj.mj_step(model, data, decimation)

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

        base_pos = data.qpos[:3]
        base_lin_vel2 = (base_pos - prev_base_pos) / (model.opt.timestep * decimation)
        mj.mju_rotVecQuat(base_lin_vel2, base_lin_vel2, neg_base_quat)
        prev_base_pos = base_pos.copy()

        command = np.array([lin_speed, 0, ang_speed])

        # print("command: ", command)
        # print("base_lin_vel: ", base_lin_vel)
        # print("command: ", command)

        viewer.user_scn.ngeom = 0
        # draw an arrow for env 0
        mj.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mj.mjtGeom.mjGEOM_ARROW, 
            size=np.array([0.02, 0.02, np.linalg.norm(base_lin_vel) * 5]),
            rgba=np.array([255, 255, 255, 255]), 
            pos=np.array([0, 0, 1]), 
            # mat=vec_to_rot_matrix(projected_gravity).flatten()
            mat=vec_to_rot_matrix(base_lin_vel).flatten()
        )
        mj.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.02, 0.02, np.linalg.norm(base_lin_vel2) * 5]),
            rgba=np.array([255, 255, 255, 255]),
            pos=np.array([0, 0, 2]),
            # mat=vec_to_rot_matrix(projected_gravity).flatten()
            mat=vec_to_rot_matrix(base_lin_vel2).flatten()
        )
        viewer.user_scn.ngeom = 2

        tibia_contact_forces = data.sensordata[:6].copy()
        feet_contact_forces = data.sensordata[6:12].copy()

        # make tibia contact forces zero if the corresponding foot force is not zero
        # because the tibia site contains the foot site
        tibia_contact_forces *= (feet_contact_forces == 0)
        print("tibia contact forces: ", tibia_contact_forces)
        print("feet contact forces: ", feet_contact_forces)

        # print sensor data
        # print("sensor data: ", data.sensordata)

        # qposs.append((data.time, np.array(data.qpos.copy()), np.array(data.qvel.copy()), np.array(data.act.copy())))

        # print(len(qposs))

        viewer.sync()

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