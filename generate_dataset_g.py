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

maze = Maze2D(gui=False)

env_num = 200
start_num = 500

# Generate trajectories
dataset = []
for i in range(env_num):
    print("generating traj in env {}".format(i))
    maze.clear_obstacles()

    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(i))
    with open(osp.join(data_dir, "obstacle_dict.json"), 'r') as f:
        obstacle_dict = json.load(f)
        maze.load_obstacle_dict(obstacle_dict)

    dense_G = nx.read_graphml(osp.join(data_dir, "dense_g.graphml"))
    occ_grid = np.loadtxt(osp.join(data_dir, "occ_grid.txt")).tolist()

    # sample trajectories
    for start_n in dense_G.nodes():
        if dense_G.nodes[start_n]['col']:
            continue

        for goal_n in dense_G.nodes():
            if dense_G.nodes[goal_n]['col']:
                continue

        goal_pos = state_to_numpy(dense_G.nodes[goal_n]['coords']).tolist()

        path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, None, None, None)

        # sanity check
        total_dist = 0
        if len(path_nodes) > 2:
            for i, node in enumerate(path_nodes):
                if i < len(path_nodes) - 1:
                    start_pos = state_to_numpy(dense_G.nodes[node]['coords']).tolist()
                    next_pos = state_to_numpy(dense_G.nodes[path_nodes[i + 1]]['coords']).tolist()
                    dist = utils.calc_weight_states(start_pos, next_pos)
                    total_dist += dist
            # print(total_dist, dis)
            assert np.allclose(total_dist, dis)

        if len(path_nodes) > 2:
            for i, node in enumerate(path_nodes):
                if i < len(path_nodes) - 1:
                    start_pos = state_to_numpy(dense_G.nodes[node]['coords']).tolist()
                    next_pos = state_to_numpy(dense_G.nodes[path_nodes[i + 1]]['coords']).tolist()

                    non_connectable_nodes = []
                    for n in dense_G.nodes():
                        if not dense_G.has_edge(n, path_nodes[i + 1]):
                            non_connectable_nodes.append(state_to_numpy(dense_G.nodes[n]['coords']).tolist())

                    dataset.append([start_pos, goal_pos, occ_grid, next_pos, non_connectable_nodes, dis])

                    dist = utils.calc_weight_states(start_pos, next_pos)
                    dis -= dist

            assert np.allclose(dis, 0)

with open(osp.join(CUR_DIR, "dataset/data_{}_g.json".format(args.name)), 'w') as f:
    json.dump(dataset, f)


