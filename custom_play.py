import mujoco as mj
import mujoco.viewer as mjv
from nikengine.engine import EngineNode, set_time_s, config
from pynput.keyboard import Key, Listener, Controller, KeyCode
import time

# Dictionary to store the state of the keys
lin = 0.05
ang = 0.2

lin_speed = 0.0
ang_speed = 0.0

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

model.opt.timestep = 0.002

engine = EngineNode()
engine.update(0.0, 0.0, 'idle')
config.ENGINE_FPS = 1.0 / model.opt.timestep

print("time step: ", model.opt.timestep)

accum = 0
prev = time.time()

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        set_time_s(data.time)
        actions = engine.update(lin_speed, ang_speed, 'awake', 'walk')
        data.ctrl[:] = actions

        mj.mj_step(model, data)

        # print sensor data
        print("sensor data: ", data.sensordata)

        viewer.sync()

        accum += 1
        if accum > 1000:
            now = time.time()
            print(accum / (now - prev))
            accum = 0
            prev = now