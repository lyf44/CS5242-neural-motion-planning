import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# print(sys.path)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image
import itertools
import random
import argparse

from lego.lego.bottleneck_node import helper as helper
from lego.lego.bottleneck_node import extract_bottleneck as extract_bottleneck
from env.maze_2d import Maze2D
import lego.utils as utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', default='2')
args = parser.parse_args()

CUR_DIR = osp.dirname(osp.abspath(__file__))
DATASET_DIR = osp.join(CUR_DIR, "dataset/{}".format(args.name))

# def state_to_numpy(state):
#     strlist = state.split(',')
#     val_list = [float(s) for s in strlist]
#     return np.array(val_list)

maze = Maze2D(gui=False)

env_num = 10000

for i in range(env_num):
    # save
    directory = osp.join(DATASET_DIR, "{}".format(i))
    if not osp.exists(directory):
        os.makedirs(directory)

    num_obstacles = 5 + (i // 1000)

    print("generating env: {}, num_obstacle: {}".format(i, num_obstacles))

    # env
    maze.clear_obstacles()
    maze.random_obstacles(mode=Maze2D.BOX_ONLY, num_obstacle=num_obstacles)
    occ_grid = np.array(maze.get_occupancy_grid()).reshape(50, 50)
    occ_grid_small = utils.shrink_occ_grid(occ_grid)
    obstacle_dict = maze.get_obstacle_dict()

    with open(osp.join(directory, "occ_grid.txt"), 'w') as f:
        np.savetxt(f, occ_grid_small.reshape(1, -1))
    utils.visualize_nodes(occ_grid_small, [], None, None, show=False, save=True, file_name=osp.join(directory, "occ_grid.png"))