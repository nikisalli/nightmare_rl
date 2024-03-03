import mujoco as mj
import mujoco.viewer as mjv
import time
import numpy as np
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decimation", type=int, default=4)
parser.add_argument("-e", "--env_num", type=int, default=2048)
parser.add_argument("-s", "--num_steps", type=int, default=10)
parser.add_argument("-t", "--num_threads", type=int, default=12)

args = parser.parse_args()
decimation = args.decimation
env_num = args.env_num
num_steps = args.num_steps
num_threads = args.num_threads

# model = mj.load_model_from_path("models/nightmare_v3/mjmodel.xml")
model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")
model.opt.timestep = 0.0025
data = [mj.MjData(model) for _ in range(env_num)]

def thread_fn(start, end):
    for i in range(start, end):
        data[i].ctrl[:] = np.zeros(18)
        mj.mj_step(model, data[i], decimation)

tstart = time.time()
for _ in range(num_steps):
    threads = []

    batch_size = env_num // num_threads

    for i in range(num_threads):
        start = i * batch_size
        end = (i + 1) * batch_size if i < num_threads - 1 else env_num
        threads.append(threading.Thread(target=thread_fn, args=(start, end)))
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

print(env_num * num_steps * decimation / (time.time() - tstart), "steps per second")