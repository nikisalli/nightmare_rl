import mujoco as mj
import mujoco.viewer as mjv
from nikengine.engine import EngineNode, set_time_s, config
from pynput.keyboard import Key, Listener, Controller, KeyCode
import time
import numpy as np

# Dictionary to store the state of the keys
lin = 0.05
ang = 0.2

lin_speed = 0.0
ang_speed = 0.0

num_envs = 1

def quat_to_rot_matrix(quat):
    """Convert a quaternion into a full three-dimensional rotation matrix."""
    w, x, y, z = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(float).eps:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
        [[1.0-(yY+zZ), xY-wZ, xZ+wY],
         [xY+wZ, 1.0-(xX+zZ), yZ-wX],
         [xZ-wY, yZ+wX, 1.0-(xX+yY)]])

def euler_to_quat(euler):
    """Convert a set of Euler angles into a quaternion."""
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def vec_to_rot_matrix(vec):
    """Convert a 3D vector into a full three-dimensional rotation matrix. that rotates the z-axis to the vector."""
    # Normalize the input vector
    vec = vec / np.linalg.norm(vec)

    # Compute the cross product of the vector and the z-axis
    cross = np.cross([0, 0, 1], vec)

    # Compute the dot product of the vector and the z-axis
    dot = np.dot([0, 0, 1], vec)

    # Compute the skew-symmetric cross-product matrix of cross
    cross_matrix = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])

    # Compute the rotation matrix using the formula
    rotation_matrix = np.eye(3) + cross_matrix + np.matmul(cross_matrix, cross_matrix) * ((1 - dot) / (np.linalg.norm(cross) ** 2))

    return rotation_matrix


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
data = [mj.MjData(model) for _ in range(num_envs)]

model.opt.timestep = 0.002

engine = EngineNode()
engine.update(0.0, 0.0, 'idle')
config.ENGINE_FPS = 1.0 / model.opt.timestep

print("time step: ", model.opt.timestep)

class lol:
    def __init__(self):
        self.viewer = mj.viewer.launch_passive(model, data[0])
        self.viewer.sync()
        self.data = data
        self.num_envs = num_envs

        accum = 0
        prev = time.time()

        self.gravity_vec = np.array([0., 0., -9.81])

        self.body_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'base_link')

        with self.viewer.lock():
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        while self.viewer.is_running():
            set_time_s(data[0].time)
            self.actions = engine.update(lin_speed, ang_speed, 'awake', 'walk')
            for d in self.data:
                d.ctrl[:] = self.actions

            self.base_quat = np.array([self.data[env_id].xquat[1] for env_id in range(self.num_envs)])
            print(self.data[0].qpos[:18])
            self.base_lin_vel = np.array([self.data[env_id].cvel[self.body_index][3:6] for env_id in range(self.num_envs)])
        
            rot_matrices = np.array([quat_to_rot_matrix(quat) for quat in self.base_quat])
            # Transpose the rotation matrices to get the inverse rotation
            rot_matrices = np.transpose(rot_matrices, (0, 2, 1))
            self.base_lin_vel = np.einsum('ijk,ik->ij', rot_matrices, self.base_lin_vel)

            self.projected_gravity = np.repeat(self.gravity_vec[None, :], self.num_envs, axis=0)

            self.projected_gravity = np.einsum('ijk,ik->ij', rot_matrices, self.projected_gravity)

            # draw the base velocity on env 0
            self.viewer.user_scn.ngeom = 0
            # draw an arrow for env 0
            mj.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mj.mjtGeom.mjGEOM_ARROW, 
                size=np.array([0.02, 0.02, 1]), 
                rgba=np.array([255, 255, 255, 255]), 
                pos=np.array([0, 0, 1]), 
                mat=vec_to_rot_matrix(self.projected_gravity[0]).flatten()
            )
            self.viewer.user_scn.ngeom = 1
            self.viewer.sync()

            for d in self.data:
                mj.mj_step(model, d)


            accum += 1
            if accum > 1000:
                now = time.time()
                print(accum / (now - prev))
                accum = 0
                prev = now

lol()