import mujoco as mj
import mujoco.viewer as mjv
import time
import pickle
import glob
import os
from pynput import keyboard  # pip install pynput
import numpy as np

model = mj.MjModel.from_xml_path("models/nightmare_v3/mjmodel.xml")
data = mj.MjData(model)

prevt = time.time()

# Get all .pkl files in logs directory and its subdirectories
# pkl_files = glob.glob('logs/nightmare_v3/2024-03-10 02:54:19.368795/**/*.pkl', recursive=True)
# pkl_files = glob.glob('logs/nightmare_v3/2024-03-10 04:27:30.671506/**/*.pkl', recursive=True)
# pkl_files = glob.glob('logs/nightmare_v3/2024-03-10 10:59:22.341365/**/*.pkl', recursive=True)
# pkl_files = glob.glob('logs/nightmare_v3/2024-03-10 15:13:33.092631/**/*.pkl', recursive=True)
# pkl_files = glob.glob('logs/nightmare_v3/2024-03-10 16:14:21.539763/**/*.pkl', recursive=True)
pkl_files = glob.glob('logs/nightmare_v3/2024-03-16 14:33:52.968470/**/*.pkl', recursive=True)
# Sort the file paths alphanumerically
pkl_files.sort()

# Define a flag for skipping
skip = False

# Define a listener for space key press
def on_press(key):
    global skip
    if key == keyboard.Key.space:
        skip = not skip

def vec_to_rot_matrix(vec):
    """Convert a 3D vector into a full three-dimensional rotation matrix. that rotates the z-axis to the vector."""
    vec = vec / np.linalg.norm(vec)
    cross = np.cross([0, 0, 1], vec)
    dot = np.dot([0, 0, 1], vec)
    cross_matrix = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + cross_matrix + np.matmul(cross_matrix, cross_matrix) * ((1 - dot) / (np.linalg.norm(cross) ** 2))


# Start the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Initialize an empty list to store all qposs
all_qposs = []

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        qposs = pickle.load(f)
        all_qposs.extend(qposs)

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    for i, qpos in enumerate(all_qposs):
        # delete the previous line in the terminal
        # print("\033[F", end="")
        print(i, "/", len(all_qposs))
        t, qpos, qvel, act = qpos
        data.time = t
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.act[:] = act
        mj.mj_step(model, data)
        viewer.sync()

        # for env_id in range(self.num_envs): mj.mju_rotVecQuat(self.base_lin_vel[env_id], self.data[env_id].cvel[self.body_index][3:6], self.base_quat[env_id])
        # for env_id in range(self.num_envs): mj.mju_rotVecQuat(self.base_ang_vel[env_id], self.data[env_id].cvel[self.body_index][:3], self.base_quat[env_id])

        base_quat = data.qpos[3:7]
        neg_base_quat = np.zeros_like(base_quat)
        mj.mju_negQuat(neg_base_quat, base_quat)
        gravity = np.array([0, 0, -9.81])
        projected_gravity = np.zeros_like(gravity)
        mj.mju_rotVecQuat(projected_gravity, gravity, neg_base_quat)
        base_lin_vel = data.cvel[1][3:6]
        mj.mju_rotVecQuat(base_lin_vel, data.cvel[1][3:6], neg_base_quat)
        base_ang_vel = data.cvel[1][:3]
        mj.mju_rotVecQuat(base_ang_vel, data.cvel[1][:3], neg_base_quat)

        # viewer.user_scn.ngeom = 0
        # # draw an arrow for env 0
        # mj.mjv_initGeom(
        #     viewer.user_scn.geoms[0],
        #     type=mj.mjtGeom.mjGEOM_ARROW, 
        #     size=np.array([0.02, 0.02, np.linalg.norm(base_lin_vel) * 5]),
        #     rgba=np.array([255, 255, 255, 255]), 
        #     pos=np.array([0, 0, 1]), 
        #     # mat=vec_to_rot_matrix(projected_gravity).flatten()
        #     mat=vec_to_rot_matrix(base_lin_vel).flatten()
        # )
        # viewer.user_scn.ngeom = 1

        # follow the base link with the camera
        viewer.cam.lookat[0] = data.qpos[0]
        viewer.cam.lookat[1] = data.qpos[1]
        viewer.cam.lookat[2] = data.qpos[2]

        print(base_lin_vel)

        if not viewer.is_running():
            break

        while not skip and time.time() - prevt < model.opt.timestep * 4:
            pass
        prevt = time.time()
    viewer.sync()

# Stop the listener
listener.stop()
