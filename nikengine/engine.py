import time
from numpy import sin, cos, arccos, arctan2, sqrt
import typing
from dataclasses import dataclass, field

# module imports
from nikengine.modules.logging import printlog, loglevel, pinfo, pwarn, perr, pfatal
from nikengine.modules.math import no_zero, asymmetrical_sigmoid, rotate, vmult, shortest_distance_two_segments_2d
from nikengine.modules.bezier import Bezier as bezier
import numpy as np

# from nightmare.modules.debug import plot

simulation_time = 0.0

# utils
def time_s():
    '''get rospy time in seconds'''
    # return time.time()
    return simulation_time

class MyConfig:
    def __init__(self):
        # =====================
        # === ABS CONSTANTS ===
        # =====================

        RIGHT = 1
        LEFT = 0
        PI = np.pi
        EPSILON = 1e-6


        # ===================================
        # === NIGHTMARE ENGINE PARAMETERS ===
        # ===================================

        # ROBOT DIMENSIONS
        STAND_MID_LEG_X = 26.e-2
        STAND_OUT_LEG_X = 20.e-2
        STAND_MID_LEG_Y = 0.e-2
        STAND_OUT_LEG_Y = 20.e-2
        BODY_LENGTH = 15.5e-2  # X
        BODY_MID_WIDTH = 18.6e-2  # Y
        BODY_OUT_WIDTH = 13.7e-2  # Y
        LEG_COXA_LENGTH = 6.5e-2
        LEG_FEMUR_LENGTH = 13.e-2
        LEG_TIBIA_LENGTH = 17.e-2

        DEFAULT_DIM = np.array([LEG_COXA_LENGTH, LEG_FEMUR_LENGTH, LEG_TIBIA_LENGTH])

        NUMBER_OF_LEGS = 6
        NUMBER_OF_SERVOS = 18
        NUMBER_OF_SERVOS_PER_LEG = 3
        NUMBER_OF_SENSORS = 6
        NUMBER_OF_SENSORS_PER_LEG = 1

        # HARDWARE PARAMS
        COMMUNICATION_PORT = "/dev/ttyACM0"
        SERVO_PORT = "/dev/ttyACM1"
        SENSOR_PORT = "/dev/ttyACM2"

        SENSOR_HEADER = [0x55, 0x55, 0x55, 0x55]
        # convert raw vals to kg
        LOAD_CELLS_CONVERSION_FACTOR = np.array([325334.23661089, 428671.46949402, -448192.25155393, 351489.18146571, -437940.86353348, -303112.40625357])

        # ENGINE PARAMETERS
        # distance legs must keep from each other
        LEG_KEEPOUT_RADIUS = 0.03

        STAND_HEIGHT = 10.e-2
        SIT_HEIGHT = 0.

        ENGINE_FPS = 51

        TIME_GET_UP_LEG_ADJ = 1.0
        TIME_SIT_LEG_ADJ = 1.0
        TIME_GET_UP = 2.5
        TIME_SIT = 2.5

        STEP_TIME = 1.
        STEP_HEIGHT = 5.e-2
        MIN_STEP_TIME = 1.
        MAX_STEP_LENGTH = 8.e-2

        # COMMAND LIMITS
        BODY_ROT_FILTER_VAL = 0.03
        BODY_TRASL_FILTER_VAL = 0.03
        WALK_ROT_FILTER_VAL = 0.03
        WALK_TRASL_FILTER_VAL = 0.03

        # ROBOT MOVEMENT LIMITS
        # walk settings
        # MAX_WALK_TRASL_VEL = np.array([10e-2, 8e-2, 0])  # m/s
        # MAX_WALK_ROT_VEL = np.array([0, 0, PI / 10])  # rad/s

        # MAX_WALK_TRASL_CMD_ACC = np.array([5e-2, 5e-2, 5e-2])  # m/s^2
        # MAX_WALK_ROT_CMD_ACC = np.array([PI / 10, PI / 10, PI / 10])  # rad/s^2

        # body displacement settings
        # MAX_BODY_TRASL = np.array([9e-2, 9e-2, 12e-2])  # m
        # MAX_BODY_ROT = np.array([PI / 10, PI / 10, PI / 10])  # rad

        # MAX_BODY_TRASL_CMD_VEL = np.array([20e-5, 20e-5, 20e-5])  # m/s
        # MAX_BODY_ROT_CMD_VEL = np.array([PI / 100, PI / 100, PI / 100])  # rad/s

        # MAX_BODY_TRASL_CMD_ACC = np.array([20e-6, 20e-6, 20e-6])  # m/s^2
        # MAX_BODY_ROT_CMD_ACC = np.array([PI / 1000, PI / 1000, PI / 1000])  # rad/s^2

        # max servo angle
        COXA_MAX_ANGLE = PI / 4  # 45 degrees
        COXA_MIN_ANGLE = -PI / 4  # -45 degrees
        FEMUR_MAX_ANGLE = (2 * PI) / 3  # 120 degrees
        FEMUR_MIN_ANGLE = -(2 * PI) / 3  # -120 degrees
        TIBIA_MAX_ANGLE = PI / 2  # 90 degrees
        TIBIA_MIN_ANGLE = -PI / 2  # -90 degrees

        # leg adj sequences
        DOUBLE_SEQUENCES = [[[1, 4], [3, 5], [2, 6]],
                            [[3, 6], [1, 5], [2, 4]]]

        # URDF PARAMS
        URDF_JOINT_OFFSETS = np.array([0, -1.2734, -0.7854, 0, -1.2734, -0.7854, 0, -1.2734, -0.7854, 0, -1.2734, -0.7854, 0, -1.2734, -0.7854, 0, -1.2734, -0.7854])
        JOINT_STATE_LABELS = ['leg1coxa', 'leg1femur', 'leg1tibia',
                            'leg2coxa', 'leg2femur', 'leg2tibia',
                            'leg3coxa', 'leg3femur', 'leg3tibia',
                            'leg4coxa', 'leg4femur', 'leg4tibia',
                            'leg5coxa', 'leg5femur', 'leg5tibia',
                            'leg6coxa', 'leg6femur', 'leg6tibia']
        URDF_OFFSET_DICT = dict(zip(JOINT_STATE_LABELS, URDF_JOINT_OFFSETS))
        LEG_TIPS = ["leg_1_tip", "leg_2_tip", "leg_3_tip", "leg_4_tip", "leg_5_tip", "leg_6_tip"]


        # ============================
        # === GENERATED PARAMETERS ===
        # ============================
        # LEG CLASS
        @dataclass
        class LEG():
            dim: np.ndarray
            servo_offset: np.ndarray
            side: int
            abs_offset: np.ndarray
            default_pose: np.ndarray
            servo_ang: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))


        legs = [
            LEG(  # leg 1
                dim=DEFAULT_DIM,
                abs_offset=np.array([BODY_OUT_WIDTH / 2, BODY_LENGTH / 2, 0]),
                default_pose=np.array([STAND_OUT_LEG_X, STAND_OUT_LEG_Y, -STAND_HEIGHT]),
                side=RIGHT,
                # servo_offset=np.array([PI / 4, -0.14874564, 0])
                servo_offset=np.array([PI / 4, 0, 0])
            ),
            LEG(  # leg 2
                dim=DEFAULT_DIM,
                abs_offset=np.array([BODY_MID_WIDTH / 2, 0, 0]),
                default_pose=np.array([STAND_MID_LEG_X, STAND_MID_LEG_Y, -STAND_HEIGHT]),
                side=RIGHT,
                # servo_offset=np.array([0, -0.14874564, 0])
                servo_offset=np.array([0, 0, 0])
            ),
            LEG(  # leg 3
                dim=DEFAULT_DIM,
                abs_offset=np.array([BODY_OUT_WIDTH / 2, -BODY_LENGTH / 2, 0]),
                default_pose=np.array([STAND_OUT_LEG_X, -STAND_OUT_LEG_Y, -STAND_HEIGHT]),
                side=RIGHT,
                # servo_offset=np.array([-PI / 4, -0.14874564, 0])
                servo_offset=np.array([-PI / 4, 0, 0])
            ),
            LEG(  # leg 4
                dim=DEFAULT_DIM,
                abs_offset=np.array([-BODY_OUT_WIDTH / 2, -BODY_LENGTH / 2, 0]),
                default_pose=np.array([-STAND_OUT_LEG_X, -STAND_OUT_LEG_Y, -STAND_HEIGHT]),
                side=LEFT,
                # servo_offset=np.array([PI / 4, -0.14874564, 0])
                servo_offset=np.array([PI / 4, 0, 0])
            ),
            LEG(  # leg 5
                dim=DEFAULT_DIM,
                abs_offset=np.array([-BODY_MID_WIDTH / 2, 0, 0]),
                default_pose=np.array([-STAND_MID_LEG_X, STAND_MID_LEG_Y, -STAND_HEIGHT]),
                side=LEFT,
                # servo_offset=np.array([0, -0.14874564, 0])
                servo_offset=np.array([0, 0, 0])
            ),
            LEG(  # leg 6
                dim=DEFAULT_DIM,
                abs_offset=np.array([-BODY_OUT_WIDTH / 2, BODY_LENGTH / 2, 0]),
                default_pose=np.array([-STAND_OUT_LEG_X, STAND_OUT_LEG_Y, -STAND_HEIGHT]),
                side=LEFT,
                # servo_offset=np.array([-PI / 4, -0.14874564, 0])
                servo_offset=np.array([-PI / 4, 0, 0])
            )
        ]

        DEFAULT_POSE: np.ndarray(shape=(6, 3)) = np.array([leg.default_pose for leg in legs])
        DEFAULT_SIT_POSE: np.ndarray(shape=(6, 3)) = np.array([[pos[0], pos[1], SIT_HEIGHT] for pos in DEFAULT_POSE.copy()])
        SERVO_OFFSET: np.ndarray(shape=(18,)) = np.array([leg.servo_offset for leg in legs]).ravel()
        POSE_OFFSET: np.ndarray(shape=(6, 3)) = np.array([leg.abs_offset for leg in legs])
        POSE_REL_CONVERT: np.ndarray(shape=(6, 3)) = np.array([[1 if leg.side else -1, 1 if leg.side else -1, 1] for leg in legs])
        SERVO_IDS: np.ndarray(shape=(18,)) = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # make this poses immutable for safety
        DEFAULT_POSE.flags.writeable = False
        DEFAULT_SIT_POSE.flags.writeable = False
        SERVO_OFFSET.flags.writeable = False
        POSE_OFFSET.flags.writeable = False
        POSE_REL_CONVERT.flags.writeable = False
        SERVO_IDS.flags.writeable = False

        # GAITS
        GAIT = {'tripod': [np.array([True, False, True, False, True, False]),
                        np.array([False, True, False, True, False, True])],
                'ripple': [np.array([True, False, False, False, True, False]),
                        np.array([False, True, False, True, False, False]),
                        np.array([False, False, True, False, False, True])],
                'wave': [np.array([True, False, False, False, False, False]),
                        np.array([False, True, False, False, False, False]),
                        np.array([False, False, True, False, False, False]),
                        np.array([False, False, False, True, False, False]),
                        np.array([False, False, False, False, True, False]),
                        np.array([False, False, False, False, False, True])]}

        self.RIGHT = RIGHT
        self.LEFT = LEFT
        self.PI = PI
        self.EPSILON = EPSILON
        self.STAND_MID_LEG_X = STAND_MID_LEG_X
        self.STAND_OUT_LEG_X = STAND_OUT_LEG_X
        self.STAND_MID_LEG_Y = STAND_MID_LEG_Y
        self.STAND_OUT_LEG_Y = STAND_OUT_LEG_Y
        self.BODY_LENGTH = BODY_LENGTH
        self.BODY_MID_WIDTH = BODY_MID_WIDTH
        self.BODY_OUT_WIDTH = BODY_OUT_WIDTH
        self.LEG_COXA_LENGTH = LEG_COXA_LENGTH
        self.LEG_FEMUR_LENGTH = LEG_FEMUR_LENGTH
        self.LEG_TIBIA_LENGTH = LEG_TIBIA_LENGTH
        self.DEFAULT_DIM = DEFAULT_DIM
        self.NUMBER_OF_LEGS = NUMBER_OF_LEGS
        self.NUMBER_OF_SERVOS = NUMBER_OF_SERVOS
        self.NUMBER_OF_SERVOS_PER_LEG = NUMBER_OF_SERVOS_PER_LEG
        self.NUMBER_OF_SENSORS = NUMBER_OF_SENSORS
        self.NUMBER_OF_SENSORS_PER_LEG = NUMBER_OF_SENSORS_PER_LEG
        self.COMMUNICATION_PORT = COMMUNICATION_PORT
        self.SERVO_PORT = SERVO_PORT
        self.SENSOR_PORT = SENSOR_PORT
        self.SENSOR_HEADER = SENSOR_HEADER
        self.LOAD_CELLS_CONVERSION_FACTOR = LOAD_CELLS_CONVERSION_FACTOR
        self.LEG_KEEPOUT_RADIUS = LEG_KEEPOUT_RADIUS
        self.STAND_HEIGHT = STAND_HEIGHT
        self.SIT_HEIGHT = SIT_HEIGHT
        self.ENGINE_FPS = ENGINE_FPS
        self.TIME_GET_UP_LEG_ADJ = TIME_GET_UP_LEG_ADJ
        self.TIME_SIT_LEG_ADJ = TIME_SIT_LEG_ADJ
        self.TIME_GET_UP = TIME_GET_UP
        self.TIME_SIT = TIME_SIT
        self.STEP_TIME = STEP_TIME
        self.STEP_HEIGHT = STEP_HEIGHT
        self.MIN_STEP_TIME = MIN_STEP_TIME
        self.MAX_STEP_LENGTH = MAX_STEP_LENGTH
        self.BODY_ROT_FILTER_VAL = BODY_ROT_FILTER_VAL
        self.BODY_TRASL_FILTER_VAL = BODY_TRASL_FILTER_VAL
        self.WALK_ROT_FILTER_VAL = WALK_ROT_FILTER_VAL
        self.WALK_TRASL_FILTER_VAL = WALK_TRASL_FILTER_VAL
        self.COXA_MAX_ANGLE = COXA_MAX_ANGLE
        self.COXA_MIN_ANGLE = COXA_MIN_ANGLE
        self.FEMUR_MAX_ANGLE = FEMUR_MAX_ANGLE
        self.FEMUR_MIN_ANGLE = FEMUR_MIN_ANGLE
        self.TIBIA_MAX_ANGLE = TIBIA_MAX_ANGLE
        self.TIBIA_MIN_ANGLE = TIBIA_MIN_ANGLE
        self.DOUBLE_SEQUENCES = DOUBLE_SEQUENCES
        self.URDF_JOINT_OFFSETS = URDF_JOINT_OFFSETS
        self.JOINT_STATE_LABELS = JOINT_STATE_LABELS
        self.URDF_OFFSET_DICT = URDF_OFFSET_DICT
        self.LEG_TIPS = LEG_TIPS
        self.legs = legs
        self.DEFAULT_POSE = DEFAULT_POSE
        self.DEFAULT_SIT_POSE = DEFAULT_SIT_POSE
        self.SERVO_OFFSET = SERVO_OFFSET
        self.POSE_OFFSET = POSE_OFFSET
        self.POSE_REL_CONVERT = POSE_REL_CONVERT
        self.SERVO_IDS = SERVO_IDS
        self.GAIT = GAIT


config = MyConfig()

EPSILON = config.EPSILON
PI = config.PI

class Command:
    state: str = 'idle'
    mode: str = 'stand'
    gait: str = 'tripod'
    body_rot: np.ndarray(shape=(3,)) = np.zeros(shape=3)
    body_trasl: np.ndarray(shape=(3,)) = np.zeros(shape=3)
    walk_rot: np.ndarray(shape=(3,)) = np.zeros(shape=3)
    walk_trasl: np.ndarray(shape=(3,)) = np.zeros(shape=3)

    def __str__(self):
        return f'''Command:\n\tstate: {self.state}\n\tmode: {self.mode}\n\tgait: {self.gait}\n\tbody_rot: {self.body_rot}\n\tbody_trasl: {self.body_trasl}\n\twalk_rot: {self.walk_rot}\n\twalk_trasl: {self.walk_trasl}'''


# data classes
class Pose:
    body_pos: np.ndarray(shape=(6, 3))
    enables: np.ndarray(shape=6)

    def __init__(self, body_pos: np.ndarray(shape=(6, 3)),
                 enables: np.ndarray(shape=6)):
        self.body_pos = body_pos.copy()
        self.enables = enables.copy()

    def __str__(self):
        return f'Pose(body_pos={self.body_pos}, enables={self.enables})'

    def __add__(self, other):
        if isinstance(other, Pose):
            if self.enables != other.enables:
                print('Warning: poses have different enables! using enables of first pose')
            return Pose(self.body_pos + other.body_pos, self.enables)
        elif isinstance(other, np.ndarray):
            if other.shape == (6, 3):
                return Pose(self.body_pos + other, self.enables)
            elif other.shape == (6,):
                # add to the z coordinate
                return Pose(self.body_pos + np.c_[np.zeros(6), np.zeros(6), other], self.enables)
            else:
                raise ValueError('other must be a Pose or a numpy array of shape (6, 3) or (6,) for z coordinate sum')
        else:
            raise ValueError('other must be a Pose or a numpy array of shape (6, 3) or (6,) for z coordinate sum')

    def __sub__(self, other):
        if isinstance(other, Pose):
            if self.enables != other.enables:
                print('Warning: poses have different enables! using enables of first pose')
            return Pose(self.body_pos - other.body_pos, self.enables)
        elif isinstance(other, np.ndarray):
            if other.shape == (6, 3):
                return Pose(self.body_pos - other, self.enables)
            elif other.shape == (6,):
                # add to the z coordinate
                return Pose(self.body_pos - np.c_[np.zeros(6), np.zeros(6), other], self.enables)
            else:
                raise ValueError('other must be a Pose or a numpy array of shape (6, 3) or (6,) for z coordinate sum')
        else:
            raise ValueError('other must be a Pose or a numpy array of shape (6, 3) or (6,) for z coordinate sum')

    def is_near(self, other) -> bool:
        '''check if poses are near'''
        return np.all(np.abs(self.body_pos - other.body_pos) < 0.01)

    def rotate(self, euler_angles: np.ndarray(shape=(3,)),
               mask: np.ndarray(shape=(6,)) = np.full((6,), True),
               pivot: np.ndarray(shape=(3,)) = None,
               inverse: bool = False):
        '''rotate the pose around a pivot point'''
        # self.body_pos = rotate(self.body_pos, euler_angles, pivot, inverse, mask)
        temp = self.body_pos.copy()
        self.body_pos = vmult(rotate(self.body_pos, euler_angles, pivot, inverse), mask) + vmult(temp, np.invert(mask))
        return self

    def translate(self, translation: np.ndarray(shape=(3,)),
                  mask: np.ndarray(shape=(6,)) = np.full((6,), True)):
        '''translate the pose'''
        # self.body_pos += translation
        self.body_pos += vmult(np.tile(translation, (6, 1)), mask)  # sum only non masked vectors
        return self

    def bezier(self, t: float,
               poses: list,
               mask: np.ndarray(shape=(6,)) = np.full((6,), True)):
        '''move every leg along a bezier curve interpolated on multiple poses
        t: parameter in [0, 1]
        poses: list of poses to interpolate between
        mask: mask to apply to not interpolate certain legs

        Note: this does not work with disabled legs, every leg must be enabled
        '''
        points = [p.body_pos for p in poses]
        temp = self.body_pos.copy()
        self.body_pos = vmult(bezier.Point(t, points), mask) + vmult(temp, np.invert(mask))
        return self

    def mask(self, pose,
             mask: np.ndarray(shape=(6,))):
        '''mask the pose with another pose'''
        self.body_pos = vmult(pose.body_pos, mask) + vmult(self.body_pos, np.invert(mask))
        return self

    def copy(self):
        return Pose(self.body_pos.copy(), self.enables.copy())

    def disable(self):
        self.enables = np.zeros(6)
        return self


class RobotState:
    pose: Pose = Pose(np.zeros(shape=(6, 3)), np.ones(shape=6))
    cmd: Command = Command()
    force_sensors: np.ndarray(shape=6) = np.zeros(shape=6)
    force_sensors_raw: np.ndarray(shape=6) = np.zeros(shape=6)

    def __str__(self):
        return f'''RobotState:\n\tpose: {self.pose}\n
                   \tforce_sensors: {self.force_sensors}'''


# FSM states
class IdleState:
    def __init__(self, state: RobotState, pose: Pose):
        pass

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        if state.cmd.state == 'idle':
            return self, state.pose.disable()
        elif state.cmd.state == 'awake':
            return AdjustGetUpState(state, pose), state.pose.disable()
        else:
            perr(f'unhandled state in idle state: {state}')


class AdjustGetUpState:
    _start_pose: Pose
    _start_time: float

    def __init__(self, state: RobotState, pose: Pose):
        self._start_pose = state.pose
        self._start_time = time_s()
        # print("leg adj start pose:\n", self._start_pose)

    def task_time_s(self) -> float:
        return time_s() - self._start_time

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        advancement = self.task_time_s() / config.TIME_GET_UP_LEG_ADJ
        if advancement < 1:
            return self, Pose(self._start_pose.body_pos + (config.DEFAULT_SIT_POSE - self._start_pose.body_pos) * advancement,
                              np.ones(shape=6))  # enable all servos
        elif advancement < 2:  # wait for a bit
            return self, Pose(config.DEFAULT_SIT_POSE, np.ones(shape=6))
        else:
            # calibrate force sensors
            pinfo(f'calibrating force sensors...')
            return GetUpState(state, pose), Pose(config.DEFAULT_SIT_POSE, np.ones(shape=6))


class GetUpState:
    _start_time: float

    def __init__(self, state: RobotState, pose: Pose):
        self._start_time = time_s()
        # print("get up start pose:\n", state.pose)

    def task_time_s(self) -> float:
        return time_s() - self._start_time

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        advancement = self.task_time_s() / config.TIME_GET_UP
        if advancement < 1:
            return self, Pose(config.DEFAULT_SIT_POSE + (config.DEFAULT_POSE - config.DEFAULT_SIT_POSE) * asymmetrical_sigmoid(advancement), np.ones(shape=6))
        elif advancement > 1 and state.cmd.state == 'idle':
            return AdjustSitState(state, pose), Pose(config.DEFAULT_POSE, np.ones(shape=6))
        elif advancement > 1 and state.cmd.state == 'awake' and state.cmd.mode == 'stand':
            return StandState(state, pose), Pose(config.DEFAULT_POSE, np.ones(shape=6))
        elif advancement > 1 and state.cmd.state == 'awake' and state.cmd.mode == 'walk':
            return WalkState(state, pose), Pose(config.DEFAULT_POSE, np.ones(shape=6))
        else:
            perr(f'unhandled state in get up state: {state}')


class SitState:
    _start_time: float

    def __init__(self, state: RobotState, pose: Pose):
        self._start_time = time_s()

    def task_time_s(self) -> float:
        return time_s() - self._start_time

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        advancement = self.task_time_s() / config.TIME_SIT
        if advancement < 1:
            return self, Pose(config.DEFAULT_POSE + (config.DEFAULT_SIT_POSE - config.DEFAULT_POSE) * asymmetrical_sigmoid(advancement), np.ones(shape=6))
        else:
            return IdleState(state, pose), Pose(config.DEFAULT_SIT_POSE, np.ones(shape=6))


class AdjustSitState:
    _start_time: float

    def __init__(self, state: RobotState, pose: Pose):
        self._start_time = time_s()

    def task_time_s(self) -> float:
        return time_s() - self._start_time

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        return SitState(state, pose), Pose(config.DEFAULT_POSE, np.ones(shape=6))


class StandState:
    _start_pose: Pose

    def __init__(self, state: RobotState, pose: Pose):
        self._start_pose = pose.copy()
        # print("stand start pose:\n", self._start_pose)

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        if state.cmd.state == 'awake' and state.cmd.mode == 'stand':
            adapt_threshold = 1.0
            adapt_gain = 0.05
            min_z_offset = -0.05
            max_z_offset = 0.05

            # z_offset = (state.force_sensors > adapt_threshold) * (state.force_sensors - adapt_threshold) * adapt_gain
            z_offset = np.zeros(6)
            # z_offset[5] = state.force_sensors[5] * adapt_gain

            # print(f'z_offset: {z_offset}')
            return self, self._start_pose.copy().translate(state.cmd.body_trasl).rotate(state.cmd.body_rot) + z_offset
        elif state.cmd.state == 'awake' and state.cmd.mode == 'walk':
            return WalkState(state, pose), pose.copy()  # return last translated and rotated pose
        elif state.cmd.state == 'idle':
            return AdjustSitState(state, pose), pose.copy()
        else:
            perr(f'unhandled state in stand state: {state}')


class WalkState:
    _gait_step: int  # step index from 0 to len(gait) - 1
    _gait_step_state: float  # 0 to 1
    _last_step_pose: Pose  # pose of the last step

    def __init__(self, state: RobotState, pose: Pose):
        self._gait_step = 0
        self._gait_step_state = 0
        self._last_step_pose = pose.copy()
        self._gait = config.GAIT[state.cmd.gait]

    def update(self, state: RobotState, pose: Pose) -> typing.Tuple[typing.Any, Pose]:
        if (state.cmd.state == 'awake' and state.cmd.mode == 'walk') or self._gait_step_state != 0:
            # translate and rotate the current pose according to the walk command
            temp = pose.copy()
            # calculate max speed for the current step
            # get current step
            gait_step = self._gait[self._gait_step]

            # calculate reduction factor
            def cost(x: float) -> float:
                local_temp = temp.copy()
                # ##### PREDICTION
                # predict robot pose at the end of the step
                # legs on ground
                legs_on_ground_mask = np.invert(gait_step)
                total_mult_factor = x * 2 * len(self._gait) * (1 - self._gait_step_state)
                local_temp.translate(- state.cmd.walk_trasl * total_mult_factor, legs_on_ground_mask).rotate(- state.cmd.walk_rot * total_mult_factor, legs_on_ground_mask)
                # stepping legs
                legs_stepping_mask = gait_step
                total_mult_factor_step = x * config.STEP_TIME
                local_temp.mask(Pose(config.DEFAULT_POSE, np.ones((6,))).translate(state.cmd.walk_trasl * total_mult_factor_step).rotate(state.cmd.walk_rot * total_mult_factor_step), legs_stepping_mask)
                # calculate the distance from every leg tip
                dists = []
                for i in range(6):
                    for j in range(6):
                        if i != j:
                            # remove z component
                            p1a = local_temp.body_pos[i][:2]
                            p1b = config.POSE_OFFSET[i][:2]
                            p2a = local_temp.body_pos[j][:2]
                            p2b = config.POSE_OFFSET[j][:2]
                            min_segment_dist = shortest_distance_two_segments_2d(p1a, p1b, p2a, p2b)
                            dists.append(min_segment_dist)
                # get min distance
                dist = config.LEG_KEEPOUT_RADIUS - min(dists)
                if dist < 0:
                    return 0
                else:
                    return dist

            # ##### OPTIMIZATION
            red = 1  # default reduction factor
            # opt_steps = []
            for i in range(10):
                current_cost = cost(red)
                if current_cost < 0.01:
                    break
                if red < 0:
                    break
                # update red
                red -= 0.1
                # opt_steps.append([red, current_cost])

            # print(red)

            # debug optimization plot
            # reds = np.linspace(0, 2, 20)
            # costs = [cost(red) for red in reds]
            # plot(reds, costs, opt_steps)

            # translate and rotate legs on ground
            legs_on_ground_mask = np.invert(gait_step)
            total_mult_factor = red * (1 / config.ENGINE_FPS) * 2 * len(self._gait)
            # the minus sign is because the legs on ground need to move in the opposite direction to move the robot forward
            temp.translate(- state.cmd.walk_trasl * total_mult_factor, legs_on_ground_mask).rotate(- state.cmd.walk_rot * total_mult_factor, legs_on_ground_mask)

            # move along step spline if legs not on ground
            legs_stepping_mask = gait_step
            # ahead of time step prediction
            total_mult_factor_step = red * config.STEP_TIME
            target_pose = Pose(config.DEFAULT_POSE, np.ones((6,))).translate(state.cmd.walk_trasl * total_mult_factor_step).rotate(state.cmd.walk_rot * total_mult_factor_step)
            points = [
                self._last_step_pose.copy(),
                self._last_step_pose.copy().translate(np.array([0, 0, config.STEP_HEIGHT])),
                target_pose.copy().translate(np.array([0, 0, config.STEP_HEIGHT])),
                target_pose.copy()
            ]
            temp.bezier(self._gait_step_state, points, legs_stepping_mask)

            # update step state when a substep is completed
            self._gait_step_state += len(self._gait) / (config.STEP_TIME * config.ENGINE_FPS)
            if self._gait_step_state > 1:
                self._gait = config.GAIT[state.cmd.gait]
                self._gait_step_state = 0
                self._gait_step = (self._gait_step + 1) % len(self._gait)
                self._last_step_pose = pose.copy()

            return self, temp
        elif state.cmd.state == 'awake' and state.cmd.mode == 'stand':
            return StandState(state, pose), pose.copy()
        elif state.cmd.state == 'idle':
            return IdleState(state, pose), Pose(config.DEFAULT_POSE, np.ones(shape=6))
        else:
            perr(f'unhandled state in walk state: {state}')


class FiniteStateMachine:
    # current fsm state instance
    _state: typing.Any
    # last commanded pose by the fsm
    _pose: Pose

    def __init__(self, state: RobotState):
        self._state = IdleState(state, state.pose)
        self._pose = state.pose

    def update(self, state: RobotState) -> Pose:
        '''get next state and updated pose'''
        self._state, pose = self._state.update(state, self._pose)
        self._pose = pose.copy()
        # print(state.force_sensors)
        # print("current_state:", self._state.identifier)
        return pose


class EngineNode:
    _robot_state: RobotState
    _state_machine: FiniteStateMachine

    def __init__(self):
        # hardware robot state constantly updated by the callbacks
        self._robot_state = RobotState()
        self._robot_state.cmd.body_rot = np.zeros(shape=3)
        self._robot_state.cmd.body_trasl = np.zeros(shape=3)
        self._robot_state.cmd.walk_rot = np.zeros(shape=3)
        self._robot_state.cmd.walk_trasl = np.zeros(shape=3)

        # state machine
        self._finite_state_machine = FiniteStateMachine(self._robot_state)

        self._robot_state.pose.body_pos = config.DEFAULT_POSE.copy()

        pinfo("ready")

    @staticmethod
    def relative_ik(rel_pos, leg_dim: np.ndarray(shape=(6, 3))) -> np.ndarray(shape=(3,)):
        x, y, z = rel_pos
        CX, FM, TB = leg_dim

        # position validity check
        coxa_director = np.array([x, y, 0]) / sqrt(x ** 2 + y ** 2)
        coxa_tip = coxa_director * CX
        tip_to_coxa_dist = np.linalg.norm(rel_pos - coxa_tip)
        director = (rel_pos - coxa_tip) / tip_to_coxa_dist
        if tip_to_coxa_dist > FM + TB:
            pwarn("position is not reachable! (too far) finding a possible one...")
            x, y, z = coxa_tip + (FM + TB - EPSILON) * director
        elif tip_to_coxa_dist < abs(FM - TB):
            pwarn("position is not reachable! (too close) finding a possible one...")
            x, y, z = coxa_tip + (abs(FM - TB) + EPSILON) * director

        d1 = sqrt(y**2 + x**2) - CX
        d = sqrt(z**2 + (d1)**2)
        alpha = -arctan2(y, x)
        beta = arccos((z**2 + d**2 - d1**2) / (2 * (- no_zero(z)) * d)) + arccos((FM**2 + d**2 - TB**2) / (2 * FM * d))
        gamma = - arccos((FM**2 + TB**2 - d**2) / (2 * FM * TB)) + 2 * PI
        return np.array([alpha, beta - PI / 2, gamma - (PI / 2) * 3])

    def set_hardware_pose(self, pose: Pose) -> np.ndarray(shape=(18,)):
        '''compute ik and write angles to hardware with current enables'''
        rel_poses = (pose.body_pos - config.POSE_OFFSET) * config.POSE_REL_CONVERT
        angles = np.ravel(np.array([self.relative_ik(rel, leg.dim) for rel, leg in zip(rel_poses, config.legs)])) + config.SERVO_OFFSET
        # publish the angles
        return angles

    def update(self, lin_speed, ang_speed, state, mode='stand'):
        self._robot_state.cmd.walk_trasl = np.array([0, lin_speed, 0])
        self._robot_state.cmd.walk_rot = np.array([0, 0, ang_speed])
        self._robot_state.cmd.state = state
        self._robot_state.cmd.mode = mode
        return self.set_hardware_pose(self._finite_state_machine.update(self._robot_state)) + config.URDF_JOINT_OFFSETS

def set_time_s(time):
    global simulation_time
    simulation_time = time