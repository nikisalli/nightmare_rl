import mujoco as mj
import mujoco.viewer as mjv
import time
import pickle

model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")

with open("logs/nightmare_v3/2024-03-03 14:05:30.308070/1709475716.pkl", "rb") as f:
    qposs = pickle.load(f)

data = mj.MjData(model)

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    for t, qpos, qvel, act in qposs:
        data.time = t
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.act[:] = act
        mj.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    viewer.sync()