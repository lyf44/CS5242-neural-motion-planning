import os.path as osp
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image
import itertools
import random
import argparse

from env.maze_2d import Maze2D
import utils
import astar

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='3')
# parser.add_argument('--goal', action='store_true', default=False)
args = parser.parse_args()

def state_to_numpy(state):
    strlist = state.split(',')
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

maze = Maze2D()

env_num = 500
path_per_env = 1000

# Generate trajectories
dataset = []
for i in range(env_num):
    print("generating paths in env {}".format(i))
    maze.clear_obstacles()

    data_dir = "./dataset/{}".format(i)
    with open(osp.join(data_dir, "obstacle_dict.json"), 'r') as f:
        obstacle_dict = json.load(f)
        maze.load_obstacle_dict(obstacle_dict)

    dense_G = nx.read_graphml(osp.join(data_dir, "dense_g.graphml"))
    occ_grid = np.loadtxt(osp.join(data_dir, "occ_grid.txt")).tolist()

    path_num = 0
    # sample trajectories
    for start_n in dense_G.nodes():
        if dense_G.nodes[start_n]['col']:
            continue

        for goal_n in dense_G.nodes():
            if dense_G.nodes[goal_n]['col']:
                continue

            start_pos = utils.state_to_numpy(dense_G.nodes[start_n]['coords']).tolist()
            goal_pos = utils.state_to_numpy(dense_G.nodes[goal_n]['coords']).tolist()
            path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, None, None, None)

            if len(path_nodes) > 2:
                path = []
                for i, node in enumerate(path_nodes):
                    node_pos = utils.state_to_numpy(dense_G.nodes[node]['coords']).tolist()
                    path.append(node_pos)

                dataset.append([start_pos, goal_pos, occ_grid, path])
                path_num += 1

            if path_num >= path_per_env:
                break

        if path_num >= path_per_env:
            break

    print(len(dataset))

with open(osp.join(CUR_DIR, "dataset/data_path_2.json"), 'w') as f:
    json.dump(dataset, f)

print('Generating waypoints')

with open(osp.join(CUR_DIR, "dataset/data_path_2.json"), 'r') as _file:
    data_path = json.load(_file)

dataset_waypoint = []
for data_point in data_path:
    start_pos, goal_pos, occ_grid, path = data_point
    for i in range(1, len(path)):
        prev_pos = path[i - 1]
        current_pos = path[i]
        dataset_waypoint.append([prev_pos, goal_pos, occ_grid, current_pos])

with open(osp.join(CUR_DIR, "dataset/data_waypoints_2.json"), 'w') as f:
    json.dump(dataset_waypoint, f)

