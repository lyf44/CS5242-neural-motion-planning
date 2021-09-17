import pybullet as p
from scipy.spatial.transform import Rotation as R
import math

from pb_ompl import PbOMPLRobot

class MyPlanarRobot(PbOMPLRobot):
    def __init__(self, id, base_xy_bounds=5.0) -> None:
        self.id = id
        self.num_dim = 4
        self.joint_idx=[0,1]
        self.reset()

        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y
        # self.joint_bounds.append([math.radians(-180), math.radians(180)]) # theta
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_0
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_1

    def get_joint_bounds(self):
        return self.joint_bounds

    def get_joint_lower_bounds(self):
        robot_bounds_low = [bound[0] for bound in self.joint_bounds]
        return robot_bounds_low

    def get_joint_higher_bounds(self):
        robot_bounds_high = [bound[1] for bound in self.joint_bounds]
        return robot_bounds_high

    def get_cur_state(self):
        return self.state

    def set_state(self, state):
        pos = [state[0], state[1], 0]
        r = R.from_euler('z', 0)
        quat = r.as_quat()
        p.resetBasePositionAndOrientation(self.id, pos, quat)
        self._set_joint_positions(self.joint_idx, state[2:])

        self.state = state

    def reset(self):
        p.resetBasePositionAndOrientation(self.id, [0,0,0], [0,0,0,1])
        self._set_joint_positions(self.joint_idx, [0,0])
        self.state = [0] * self.num_dim

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)