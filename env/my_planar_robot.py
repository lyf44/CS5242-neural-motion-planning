class MyPlanarRobot():
    def __init__(self, base_xy_bounds=5.0) -> None:
        self.num_dim = 2
        self.joint_idx=[0,1]
        self.reset()

        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y

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
        self.state = state

    def reset(self):
        self.state = [0] * self.num_dim

