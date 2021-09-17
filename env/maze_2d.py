import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import math
import random
import json
import sys
import os.path as osp

from my_planar_robot import MyPlanarRobot
import pb_ompl
# from config import ROOT_DI

ROOT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "../")
DATASET_DIR = osp.join(ROOT_DIR, "dataset")

# -------------- Settings ----------------
RANDOM = True
TOTAL_START_GOAL_CNT = 50
MAZE_SIZE = 5
OCC_GRID_RESOLUTION = 0.1

class Maze2D():
    EMPTY = 0
    GAP_ONLY = 1
    BOX_ONLY = 2
    GAP_AND_BOX = 3

    def __init__(self, gui=True):
        self.obstacles = []

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load floor
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        # load robot
        robot_model_path = osp.join(ROOT_DIR, "my_planar_robot_model/urdf/my_planar_robot.xacro")
        robot_id = p.loadURDF(robot_model_path, (0,0,0))
        robot = MyPlanarRobot(robot_id, base_xy_bounds = MAZE_SIZE / 2.0)
        self.robot = robot

        # 2d occupancy grid
        self.occ_grid_size = int(MAZE_SIZE / OCC_GRID_RESOLUTION)
        self.occ_grid = np.zeros((1, self.occ_grid_size, self.occ_grid_size), dtype=np.uint8)
        self.small_occ_grid_size = 10

        # add surrounding walls
        half_size = MAZE_SIZE / 2.0
        # add wall
        # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 5, 1])
        # wall1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[5, 0, 1])
        # wall2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[-5, 0, 1])
        wall1 = self.add_box([half_size + 0.1, 0, 1], [0.1, half_size, 1])
        wall2 = self.add_box([-half_size - 0.1, 0, 1], [0.1, half_size, 1])
        # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 0.1, 1])
        # wall3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, 5, 1])
        # wall4 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, -5, 1])
        wall3 = self.add_box([0, half_size + 0.1, 1], [half_size, 0.1, 1])
        wall4 = self.add_box([0, -half_size - 0.1, 1], [half_size, 0.1, 1])
        self.default_obstacles = [wall1, wall2, wall3, wall4]

        # set up pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("RRTDof")
        self.pb_ompl_interface.planner.setRange(10)

        # self.pb_ompl_interface_2 = pb_ompl.PbOMPL(self.robot, self.obstacles)
        # self.pb_ompl_interface_2.set_planner("PRM")
        # self.pb_ompl_interface_2.interpolate_num = 0
        self.pb_ompl_interface_2 = pb_ompl.PbOMPLPRM(self.robot, self.obstacles)

        # internal attributes
        self.goal_robot_id = None
        self.path = None
        self.approx_path = None
        self.sg_pairs = None

        self.obstacle_dict = {}

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle not in self.default_obstacles:
                p.removeBody(obstacle)
        self.occ_grid.fill(0)
        self.obstacles = self.default_obstacles.copy()
        self.obstacle_dict = {}
        self.pb_ompl_interface.set_obstacles(self.obstacles)
        self.pb_ompl_interface_2.set_obstacles(self.obstacles)

    def random_obstacles(self, mode=GAP_AND_BOX):
        self.mode = mode
        obstacles = []

        # half_size = MAZE_SIZE / 2.0
        # # add wall
        # # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 5, 1])
        # # wall1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[5, 0, 1])
        # # wall2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[-5, 0, 1])
        # wall1 = self.add_box([half_size + 0.1, 0, 1], [0.1, half_size, 1])
        # wall2 = self.add_box([-half_size - 0.1, 0, 1], [0.1, half_size, 1])
        # # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 0.1, 1])
        # # wall3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, 5, 1])
        # # wall4 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, -5, 1])
        # wall3 = self.add_box([0, half_size + 0.1, 1], [half_size, 0.1, 1])
        # wall4 = self.add_box([0, -half_size - 0.1, 1], [half_size, 0.1, 1])

        if mode == Maze2D.GAP_ONLY or mode == Maze2D.GAP_AND_BOX:
            # random sample gap
            if RANDOM:
                vertical = random.randint(0, 1)
                if vertical:
                    x = random.randint(0, 9)
                    y = random.randint(0, 8)
                    gap1_pos = [0.5 * x - 2.25, 0.5 * y - 2]
                # gap1_pos = [random.randrange(-half_size + 1, half_size - 1), random.randrange(-half_size + 1, half_size - 1)]
                # gap2_pos_1 = random.randrange(-half_size + 0.5, half_size - 0.5)
                # while gap2_pos_1 > gap1_pos[1] - 0.5 and gap2_pos_1 < gap1_pos[1] + 0.5:
                #     gap2_pos_1 = random.randrange(-half_size + 0.5, half_size - 0.5)
                # gap2_pos = [random.randrange(-half_size + 0.5, gap1_pos[0] - 0.5), gap2_pos_1]
                # gap3_pos = [random.randrange(gap1_pos[0] + 0.5, half_size - 0.5), gap2_pos_1]
                else:
                    x = random.randint(0, 8)
                    y = random.randint(0, 9)
                    gap1_pos = [0.5 * x - 2, 0.5 * y - 2.25]
            else:
                gap1_pos = [0.5, 0.25]
                vertical = 0
                # gap2_pos = [-2, -2]
                # gap3_pos = [2, -2]

            self.add_gap(gap1_pos, vertical)

            self.obstacle_dict["gap"] = {}
            self.obstacle_dict["gap"]["pos"] = gap1_pos
            self.obstacle_dict["gap"]["vertical"] = vertical

        if mode == Maze2D.BOX_ONLY or mode == Maze2D.GAP_AND_BOX:
            # add random obstacles with boxes.
            NUM_BOX = 5
            box_positions = []
            if RANDOM:
                for _ in range(NUM_BOX):
                    # x = random.uniform(-half_size, half_size)
                    # y = random.uniform(-half_size, half_size)
                    x = random.randint(0, 9)
                    y = random.randint(0, 9)
                    x = 0.5 * x - 2.25
                    y = 0.5 * y - 2.25
                    box_positions.append((x, y))
            else:
                NUM_BOX = 5
                # box_positions.append([0.5, 0])
                # box_positions.append([-0.5, 0])
                box_positions.append([1.25, -1.25])
                box_positions.append([0.75, -1.25])
                box_positions.append([-1.25, -2.25])
                box_positions.append([0.25, 2.25])
                box_positions.append([-1.25, 1.25])

            # print(box_positions)
            for box_pos in box_positions:
                self.add_box([box_pos[0], box_pos[1], 0.5], [0.25, 0.25, 0.5])

            self.obstacle_dict["box"] = box_positions

        self.pb_ompl_interface.set_obstacles(self.obstacles)
        self.pb_ompl_interface_2.set_obstacles(self.obstacles)

    def add_gap(self, gap_pos, vertical):
        half_size = MAZE_SIZE / 2.0
        if vertical:
            wall5 = self.add_box([gap_pos[0], half_size - (half_size - 0.5 - gap_pos[1]) / 2, 1], [0.25, (half_size - 0.5 - gap_pos[1]) / 2, 1])
            wall6 = self.add_box([gap_pos[0], -half_size + (half_size - 0.5 + gap_pos[1]) / 2, 1], [0.25, (half_size - 0.5 + gap_pos[1]) / 2, 1])
        else:
            wall5 = self.add_box([half_size - (half_size - 0.5 - gap_pos[0]) / 2, gap_pos[1], 1], [(half_size - 0.5 - gap_pos[0]) / 2, 0.25, 1])
            wall6 = self.add_box([-half_size + (half_size - 0.5 + gap_pos[0]) / 2, gap_pos[1], 1], [(half_size - 0.5 + gap_pos[0]) / 2, 0.25, 1])


    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        # for occupancy grid, center is at upper left corner, unit is cm
        half_size = MAZE_SIZE / 2.0
        tmp = int(1 / OCC_GRID_RESOLUTION)
        cx = int((-box_pos[1] + half_size) * tmp)
        cy = int((box_pos[0] + half_size) * tmp)
        x_size = int(half_box_size[1] * tmp)
        y_size = int(half_box_size[0] * tmp)
        for x in range(max(0, cx - x_size), min(self.occ_grid_size, cx + x_size)):
            for y in range(max(0, cy - y_size), min(self.occ_grid_size, cy + y_size)):
                self.occ_grid[0, x, y] = 255

        self.obstacles.append(box_id)

        return box_id

    def get_occupancy_grid(self):
        return self.occ_grid

    def get_small_occupancy_grid(self):
        occ_grid_small = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                occ_grid_small[i, j] = (np.max(self.occ_grid[0, i*5:(i+1)*5, j*5:(j+1)*5]) == 255)

        return occ_grid_small

    def get_obstacle_dict(self):
        return self.obstacle_dict

    def load_obstacle_dict(self, obstacle_dict):
        if "gap" in obstacle_dict:
            self.add_gap(obstacle_dict["gap"]["pos"], obstacle_dict["gap"]["vertical"])
        if "box" in obstacle_dict:
            for box_pos in obstacle_dict["box"]:
                self.add_box([box_pos[0], box_pos[1], 0.5], [0.25, 0.25, 0.5])

        self.pb_ompl_interface.set_obstacles(self.obstacles)
        self.pb_ompl_interface_2.set_obstacles(self.obstacles)
        self.obstacle_dict = obstacle_dict

    def sample_start_goal(self, load = False):
        if load:
            print("Maze2D: loading start_goal from sg_paris.json!!!")
            with open(osp.join(ROOT_DIR, "sg_pairs.json"), 'r') as f:
                self.sg_pairs = json.load(f)

            sg = random.choice(self.sg_pairs)
            self.start = sg[0]
            self.goal = sg[1]
            # self.start = [-4,-3,0,0,0,0,0]
            # self.goal = [1,0,math.radians(-90),0,0,0,0]
            # self.goal = [-2,0,0,0,0,0,0]
        else:
            while True:
                start = [0] * self.robot.num_dim
                goal = [0] * self.robot.num_dim
                low_bounds = self.robot.get_joint_lower_bounds()
                high_bounds = self.robot.get_joint_higher_bounds()
                for i in range(self.robot.num_dim):
                    start[i] = random.uniform(low_bounds[i], high_bounds[i])
                    goal[i] = random.uniform(low_bounds[i], high_bounds[i])

                if self.pb_ompl_interface_2.is_state_valid(start) and self.pb_ompl_interface_2.is_state_valid(goal):
                    self.start = start
                    self.goal = goal
                    break

        print("Maze2d: start: {}".format(self.start))
        print("Maze2d: goal: {}".format(self.goal))

    def generate_start_goal_pairs(self, num):
        print("Maze2D: Generating {} start_goal pairs".format(num))
        cnt = 0
        self.sg_pairs = []
        while cnt <= num:
            start = [0] * self.robot.num_dim
            goal = [0] * self.robot.num_dim
            low_bounds = self.robot.get_joint_lower_bounds()
            high_bounds = self.robot.get_joint_higher_bounds()
            for i in range(self.robot.num_dim):
                start[i] = random.uniform(low_bounds[i], high_bounds[i])
                goal[i] = random.uniform(low_bounds[i], high_bounds[i])

            if self.pb_ompl_interface_2.is_state_valid(start) and self.pb_ompl_interface_2.is_state_valid(goal):
                self.sg_pairs.append([start, goal])
                cnt += 1

        with open(osp.join(ROOT_DIR, "sg_pairs.json"), 'w') as f:
            json.dump(self.sg_pairs, f)

    def plan(self, allowed_time=2.0, interpolate=False):
        self.pb_ompl_interface.clear()
        self.path = None
        self.approx_path = None

        self.robot.set_state(self.start)
        res, path = self.pb_ompl_interface.plan(self.goal, allowed_time, interpolate)
        if res:
            self.path = path
        elif path is not None:
            self.approx_path = path
        return res, path

    def plan2(self, allowed_time=2.0, interpolate=False):
        self.pb_ompl_interface_2.clear()
        self.path = None
        self.approx_path = None

        self.robot.set_state(self.start)
        res, path = self.pb_ompl_interface_2.plan(self.goal, allowed_time, interpolate)
        if res:
            self.path = path
        elif path is not None:
            self.approx_path = path
        return res, path

    def execute(self):
        if self.path is not None:
            self.pb_ompl_interface.execute(self.path)

        elif self.approx_path is not None:
            self.pb_ompl_interface.execute(self.approx_path)

    def execute2(self):
        if self.path is not None:
            self.pb_ompl_interface_2.execute(self.path)

        elif self.approx_path is not None:
            self.pb_ompl_interface_2.execute(self.approx_path)
        # for _ in range(1200):
        #     # p.stepSimulation()
        #     time.sleep(1./ 240)

        # p.disconnect()

    def visualize_goal(self, goal):
        if self.goal_robot_id is not None:
            p.removeBody(self.goal_robot_id)
        robot_model_path = osp.join(ROOT_DIR, "my_planar_robot_model/urdf/my_planar_robot_4_link.xacro")
        self.goal_robot_id = p.loadURDF(robot_model_path, (0,0,0))
        robot = MyPlanarRobot(self.goal_robot_id)
        robot.set_state(goal)

    def construct_prm(self, allowed_time=5.0, clear=True):
        if clear:
            self.pb_ompl_interface_2.clear()
        self.pb_ompl_interface_2.construct_prm(allowed_time)

if __name__ == '__main__':
    maze = Maze2D()
    maze.random_obstacles(mode=3)
    maze.sample_start_goal()
    maze.plan2(interpolate=True)
    maze.execute2()
    input("press anything")
    print("here")