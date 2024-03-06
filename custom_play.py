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

decimation = 4
action_rate = 0.08
prev_actions = np.zeros(18)

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
data = mj.MjData(model)

engine = EngineNode()
engine.update(0.0, 0.0, 'idle')
config.ENGINE_FPS = 1.0 / model.opt.timestep / decimation

print("time step: ", model.opt.timestep)

accum = 0
prev = time.time()

qposs = []

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        set_time_s(data.time)
        print("time: ", data.time)
        actions = engine.update(lin_speed, ang_speed, 'awake', 'walk')
        # actions = np.random.uniform(-0.5, 0.5, 18)
        # limit action_rate
        prev_actions += np.clip(actions - prev_actions, -action_rate, action_rate)
        kp = 20
        data.ctrl[:] = (prev_actions - data.qpos[-18:]) * kp

        mj.mj_step(model, data, decimation)

        base_quat = data.qpos[3:7]
        neg_base_quat = np.zeros_like(base_quat)
        mj.mju_negQuat(neg_base_quat, base_quat)
        gravity = np.array([0, 0, -9.81])
        projected_gravity = np.zeros_like(gravity)
        mj.mju_rotVecQuat(projected_gravity, gravity, neg_base_quat)
        print(np.sum(np.square(projected_gravity[:2])))

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