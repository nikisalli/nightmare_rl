import mujoco as mj
import mujoco.viewer as mjv
from mujoco import mjx
import time
import numpy as np
import pickle
import datetime
import os
import psutil
import jax
import jax.numpy as jnp


decimation = 8
env_num = 2048
iters = 1000000

# model = mj.load_model_from_path("models/nightmare_v3/mjmodel.xml")
model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")
model.opt.timestep = 0.0025
mjx_model = mjx.device_put(model)

@jax.vmap
def batched_step(actions):
    data = mjx.make_data(mjx_model)
    data.replace(ctrl=actions)
    for _ in range(decimation):
        mjx.step(mjx_model, data)

jit_step = jax.jit(batched_step)
print("jit done")

start = time.time()

for _ in range(iters):
    actions = jnp.zeros((env_num, 18))
    jit_step(actions)

print(env_num * iters * decimation / (time.time() - start))
