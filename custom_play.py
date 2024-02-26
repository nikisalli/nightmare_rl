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

model.opt.timestep = 0.0025

p_gain = 10
d_gain = 0.05
max_speed = 0.1
speed_p_gain = 0.05

engine = EngineNode()
engine.update(0.0, 0.0, 'idle')
config.ENGINE_FPS = 1.0 / model.opt.timestep

print("time step: ", model.opt.timestep)

accum = 0
prev = time.time()

# rolling plot of cvel[1]
import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots()
# xs = np.zeros((100, 18))
# y = np.arange(0, 100)
# # set y-axis limits
# ax.set_ylim(-10, 10)
# line1, = ax.plot(y, xs[:, 14], label='1')
# line2, = ax.plot(y, xs[:, 15], label='2')
# line3, = ax.plot(y, xs[:, 16], label='3')
# line4, = ax.plot(y, xs[:, 17], label='4')
# fig.canvas.draw()
# axbackground = fig.canvas.copy_from_bbox(ax.bbox)
# plt.show(block=False)

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        set_time_s(data.time)
        actions = engine.update(lin_speed, ang_speed, 'awake', 'walk')
        data.ctrl[:] = actions

        # fig.canvas.restore_region(axbackground)
        # xs = np.roll(xs, -1, axis=0)
        # xs[-1] = data.qvel[-18:]
        # line1.set_ydata(xs[:, 14])
        # line2.set_ydata(xs[:, 15])
        # line3.set_ydata(xs[:, 16])
        # line4.set_ydata(xs[:, 17])
        # ax.draw_artist(line1)
        # ax.draw_artist(line2)
        # ax.draw_artist(line3)
        # ax.draw_artist(line4)
        # fig.canvas.blit(ax.bbox)
        # fig.canvas.flush_events()

        # print(data.efc_force.shape)
        # print("-------------------")

        mj.mj_step(model, data)

        # viewer.sync()

        accum += 1
        if accum > 1000:
            now = time.time()
            print(accum / (now - prev))
            accum = 0
            prev = now