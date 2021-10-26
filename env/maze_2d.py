import os
import sys
import numpy as np
import math
import random
import json
import sys
import os.path as osp

from env.my_planar_robot import MyPlanarRobot
# from config import ROOT_DI

ROOT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "../")
DATASET_DIR = osp.join(ROOT_DIR, "dataset")

# -------------- Settings ----------------
RANDOM = True
TOTAL_START_GOAL_CNT = 50
MAZE_SIZE = 5
OCC_GRID_RESOLUTION = 0.1

class Maze2D():
    def __init__(self):
        self.obstacles = []

        # load robot
        robot = MyPlanarRobot(base_xy_bounds = MAZE_SIZE / 2.0)
        self.robot = robot

        # 2d occupancy grid
        self.occ_grid_size = int(MAZE_SIZE / OCC_GRID_RESOLUTION)
        self.occ_grid = np.zeros((self.occ_grid_size, self.occ_grid_size), dtype=np.uint8)
        self.small_occ_grid_size = 10

        # add surrounding walls
        half_size = MAZE_SIZE / 2.0
        # add wall
        self.add_box([half_size + 0.1, 0, 1], [0.1, half_size, 1])
        self.add_box([-half_size - 0.1, 0, 1], [0.1, half_size, 1])
        self.add_box([0, half_size + 0.1, 1], [half_size, 0.1, 1])
        self.add_box([0, -half_size - 0.1, 1], [half_size, 0.1, 1])

        # internal attributes
        self.goal_robot_id = None
        self.path = None
        self.approx_path = None
        self.sg_pairs = None

        self.obstacle_dict = {}

    def clear_obstacles(self):
        self.occ_grid.fill(0)
        self.obstacle_dict = {}
        self.inflated_occ_grid = None

    def random_obstacles(self, num_of_boxes = 8):
        # add random obstacles with boxes.
        # box_positions = [(-2.25, 2.25)]
        box_positions = []

        for _ in range(num_of_boxes):
            x = random.randint(0, 4)
            y = random.randint(0, 4)
            x = x - 2
            y = y - 2
            box_positions.append((x, y))

        # print(box_positions)
        for box_pos in box_positions:
            self.add_box([box_pos[0], box_pos[1], 0.5], [0.5, 0.5, 0.5])

        self.obstacle_dict["box"] = box_positions

        self.get_inflated_occ_grid()

    def add_box(self, box_pos, half_box_size):
        # for occupancy grid, center is at upper left corner, unit is cm
        half_size = MAZE_SIZE / 2.0
        tmp = int(1 / OCC_GRID_RESOLUTION)
        cx = (-box_pos[1] + half_size) * tmp
        cy = (box_pos[0] + half_size) * tmp
        x_size = half_box_size[1] * tmp
        y_size = half_box_size[0] * tmp
        for x in range(max(0, int(cx - x_size)), min(self.occ_grid_size, int(cx + x_size))):
            for y in range(max(0, int(cy - y_size)), min(self.occ_grid_size, int(cy + y_size))):
                self.occ_grid[x, y] = 1

    def get_occupancy_grid(self):
        return self.occ_grid

    def get_small_occupancy_grid(self):
        occ_grid_small = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(10):
                occ_grid_small[i, j] = (np.max(self.occ_grid[i*5:(i+1)*5, j*5:(j+1)*5]) == 1)

        return occ_grid_small

    def get_obstacle_dict(self):
        return self.obstacle_dict.copy()

    def load_obstacle_dict(self, obstacle_dict):
        if "box" in obstacle_dict:
            for box_pos in obstacle_dict["box"]:
                self.add_box([box_pos[0], box_pos[1], 0.5], [0.5, 0.5, 0.5])

        self.obstacle_dict = obstacle_dict

    def sample_start_goal(self):
        while True:
            start = [0] * self.robot.num_dim
            goal = [0] * self.robot.num_dim
            low_bounds = self.robot.get_joint_lower_bounds()
            high_bounds = self.robot.get_joint_higher_bounds()
            for i in range(self.robot.num_dim):
                start[i] = random.uniform(low_bounds[i], high_bounds[i])
                goal[i] = random.uniform(low_bounds[i], high_bounds[i])

            if self.is_state_valid(start) and self.is_state_valid(goal):
                self.start = start
                self.goal = goal
                break

        print("Maze2d: start: {}".format(self.start))
        print("Maze2d: goal: {}".format(self.goal))

    def get_inflated_occ_grid(self):
        if self.inflated_occ_grid is None:
            tmp = np.zeros((self.occ_grid_size + 2, self.occ_grid_size + 2), dtype=np.uint8)
            tmp[:self.occ_grid_size, :self.occ_grid_size] += self.occ_grid
            tmp[1:self.occ_grid_size + 1, :self.occ_grid_size] += self.occ_grid
            tmp[2:, :self.occ_grid_size] += self.occ_grid
            tmp[:self.occ_grid_size, 1:self.occ_grid_size+1] += self.occ_grid
            tmp[1:self.occ_grid_size + 1, 1:self.occ_grid_size+1] += self.occ_grid
            tmp[2:, 1:self.occ_grid_size+1] += self.occ_grid
            tmp[:self.occ_grid_size, 2:] += self.occ_grid
            tmp[1:self.occ_grid_size + 1, 2:] += self.occ_grid
            tmp[2:, 2:] += self.occ_grid
            tmp[tmp > 0] = 1

            self.inflated_occ_grid = tmp[1:self.occ_grid_size + 1, 1:self.occ_grid_size + 1]

    def is_state_valid(self, robot_state):
        # Inflate  for collision checking
        self.get_inflated_occ_grid()

        y, x = robot_state[0], robot_state[1]
        x = int((MAZE_SIZE / 2.0 - x) / 0.1)
        y = int((y + MAZE_SIZE / 2.0) / 0.1)

        res = (self.inflated_occ_grid[x, y] != 1)
        return res

if __name__ == '__main__':
    sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
    import utils
    import cv2

    maze = Maze2D()
    maze.random_obstacles()

    occ_grid = maze.get_occupancy_grid()
    # print(occ_grid)
    tmp = np.copy(occ_grid).reshape(50, 50, 1)
    tmp[tmp == 1] = 255
    cv2.imshow("tmp", tmp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    occ_grid = maze.inflated_occ_grid
    tmp = np.copy(occ_grid).reshape(50, 50, 1)
    tmp[tmp == 1] = 255
    cv2.imshow("tmp", tmp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    occ_grid = maze.get_small_occupancy_grid()
    print(occ_grid.shape)
    utils.visualize_nodes(occ_grid, [], None, None)

    print(maze.is_state_valid([-1.9, 2.0]))
    print(maze.is_state_valid([-1.9, 1.9]))
    print(maze.is_state_valid([-2.0, 1.9]))
    print(maze.is_state_valid([-2.0, 2.0]))