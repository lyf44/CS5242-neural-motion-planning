import os
import os.path as osp
import sys
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# print(sys.path)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image
import itertools
import random

import utils
from env.maze_2d import Maze2D


CUR_DIR = osp.dirname(osp.abspath(__file__))
DATASET_DIR = osp.join(CUR_DIR, "dataset/4")

def state_to_numpy(state):
    strlist = state.split(',')
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

maze = Maze2D(gui=False)

env_num = 100

sparse_num = 100
dense_num = 500

for i in range(env_num, env_num*2):
    # save
    directory = osp.join(DATASET_DIR, "{}".format(i))
    if not osp.exists(directory):
        os.makedirs(directory)

    # env
    maze.clear_obstacles()
    maze.random_obstacles(mode=3)
    occ_grid = np.array(maze.get_occupancy_grid()).reshape(50, 50)
    occ_grid_small = utils.shrink_occ_grid(occ_grid)
    obstacle_dict = maze.get_obstacle_dict()
    maze.sample_start_goal()
    maze.robot.set_state(maze.start)
    # continue

    # sparse graph
    # maze.construct_prm(0.5)
    # maze.pb_ompl_interface_2.get_planner_data()
    # shutil.copyfile(osp.join(CUR_DIR, "prm_graph.graphml"), osp.join(CUR_DIR, "prm_graph_sparse.graphml"))
    # sparse_G = nx.read_graphml(osp.join(CUR_DIR, "prm_graph_sparse.graphml"))
    # nodes = list(sparse_G.nodes)
    # node_pos = np.array([state_to_numpy(sparse_G.nodes[node]['coords']) for node in nodes])

    # sparse states
    low = maze.robot.get_joint_lower_bounds()
    high = maze.robot.get_joint_higher_bounds()
    states = []
    col_status = []
    for _ in range(sparse_num):
        random_state = [0] * maze.robot.num_dim
        for i in range(maze.robot.num_dim):
            random_state[i] = random.uniform(low[i], high[i])
        # if not maze.pb_ompl_interface_2.is_state_valid(random_state):
        #     collision_states.append(random_state)
        col_status.append(maze.pb_ompl_interface_2.is_state_valid(random_state))
        states.append(random_state)
    # collision_states = np.array(collision_states)
    sparse_G = nx.DiGraph()
    sparse_G.add_nodes_from([("n{}".format(i), {"coords": ','.join(map(str, state)), "col": not col_status[i]}) for i, state in enumerate(states)])

    # save
    # node_pos = np.array(states)
    node_pos = np.array([utils.state_to_numpy(sparse_G.nodes[node]['coords']) for node in sparse_G.nodes()])
    utils.visualize_nodes(occ_grid_small, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "sparse.png"))
    node_pos = np.array([utils.state_to_numpy(sparse_G.nodes[node]['coords']) for node in sparse_G.nodes() if not sparse_G.nodes[node]['col']])
    utils.visualize_nodes(occ_grid_small, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "sparse_free.png"))

    # dense graph
    # maze.construct_prm(4.5, clear=False)
    # maze.pb_ompl_interface_2.get_planner_data()
    # shutil.copyfile(osp.join(CUR_DIR, "prm_graph.graphml"), osp.join(CUR_DIR, "prm_graph_dense.graphml"))
    # dense_G = nx.read_graphml(osp.join(CUR_DIR, "prm_graph_dense.graphml"))
    # nodes = list(dense_G.nodes)
    # node_pos = np.array([state_to_numpy(dense_G.nodes[node]['coords']) for node in nodes])

    # dense states
    low = maze.robot.get_joint_lower_bounds()
    high = maze.robot.get_joint_higher_bounds()
    for _ in range(sparse_num, dense_num):
        random_state = [0] * maze.robot.num_dim
        for i in range(maze.robot.num_dim):
            random_state[i] = random.uniform(low[i], high[i])
        # if not maze.pb_ompl_interface_2.is_state_valid(random_state):
        #     collision_states.append(random_state)
        col_status.append(maze.pb_ompl_interface_2.is_state_valid(random_state))
        states.append(random_state)
    # collision_states = np.array(collision_states)
    dense_G = nx.DiGraph()
    dense_G.add_nodes_from([("n{}".format(i), {"coords": ','.join(map(str, state)), "col": not col_status[i]}) for i, state in enumerate(states)])

    # save
    # node_pos = np.array(states)
    node_pos = np.array([utils.state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes()])
    utils.visualize_nodes(occ_grid_small, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "dense.png"))
    node_pos = np.array([utils.state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes() if not dense_G.nodes[node]['col']])
    utils.visualize_nodes(occ_grid_small, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "dense_free.png"))

    print("connecting sparse graph")
    nodes = sparse_G.nodes()
    node_pairs = itertools.combinations(nodes, 2)
    # print(list(node_pairs))
    for node_pair in node_pairs:
        if not sparse_G.has_edge(node_pair[0], node_pair[1]):
            s1 = sparse_G.nodes[node_pair[0]]['coords']
            s2 = sparse_G.nodes[node_pair[1]]['coords']
            if utils.is_edge_free(maze, s1, s2):
                sparse_G.add_edge(node_pair[0], node_pair[1])
                sparse_G.add_edge(node_pair[1], node_pair[0])
    for u,v in sparse_G.edges:
        sparse_G[u][v]['weight'] = utils.calc_weight_states(sparse_G.nodes[u]['coords'], sparse_G.nodes[v]['coords'])

    print("connecting dense graph")
    nodes = dense_G.nodes()
    node_pairs = itertools.combinations(nodes, 2)
    # print(list(node_pairs))
    for node_pair in node_pairs:
        if not dense_G.has_edge(node_pair[0], node_pair[1]):
            s1 = dense_G.nodes[node_pair[0]]['coords']
            s2 = dense_G.nodes[node_pair[1]]['coords']
            if utils.is_edge_free(maze, s1, s2):
                dense_G.add_edge(node_pair[0], node_pair[1])
                dense_G.add_edge(node_pair[1], node_pair[0])
    for u,v in dense_G.edges:
        dense_G[u][v]['weight'] = utils.calc_weight_states(dense_G.nodes[u]['coords'], dense_G.nodes[v]['coords'])

    # save
    nx.write_graphml(sparse_G, osp.join(directory, "sparse_g.graphml"))
    nx.write_graphml(dense_G, osp.join(directory, "dense_g.graphml"))
    with open(osp.join(directory, "occ_grid.txt"), 'w') as f:
        np.savetxt(f, occ_grid_small.reshape(1, -1))
    with open(osp.join(directory, "obstacle_dict.json"), 'w') as f:
        json.dump(obstacle_dict, f)
    utils.visualize_nodes(occ_grid_small, [], None, None, show=False, save=True, file_name=osp.join(directory, "occ_grid.png"))

