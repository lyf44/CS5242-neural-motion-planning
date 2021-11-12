import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def state_to_numpy(state):
    strlist = state.split(',')
    # strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

def cal_edge_length(s1, s2):
    if isinstance(s1, np.ndarray):
        config1 = s1
    elif isinstance(s1, list):
        config1 = np.array(s1)
    else:
        config1 = state_to_numpy(s1)
    if isinstance(s2, np.ndarray):
        config2 = s2
    elif isinstance(s2, list):
        config2 = np.array(s2)
    else:
        config2 = state_to_numpy(s2)

    # config1 = state_to_numpy(s1)
    # config2 = state_to_numpy(s2)
    return math.sqrt(float(np.sum((config2-config1)**2)))

def is_edge_free(maze, node1_state, node2_state):
    if isinstance(node1_state, list):
        node1_pos = np.array(node1_state)
    else:
        node1_pos = state_to_numpy(node1_state)
    if isinstance(node2_state, list):
        node2_pos = np.array(node2_state)
    else:
        node2_pos = state_to_numpy(node2_state)

    diff = node2_pos - node1_pos
    edge_discretization = int(np.max(np.abs(diff)) / 0.1) + 1
    step = diff / edge_discretization
    assert(np.max(step) < 0.1 and np.min(step) > -0.1)

    for i in range(edge_discretization):
        nodepos = node1_pos + step * i
        if not maze.is_state_valid(nodepos.tolist()):
            return False
    return True

def extend(maze, v, g):
    v1 = np.array(v)
    v2 = np.array(g)
    diff = v2 - v1
    edge_discretization = int(np.max(np.abs(diff)) / 0.1) + 1
    step = diff / edge_discretization
    assert(np.max(step) < 0.1 and np.min(step) > -0.1)

    res_v = v1
    for i in range(edge_discretization):
        nodepos = v1 + step * i
        # print(nodepos)
        if not maze.is_state_valid(nodepos.tolist()):
            break

        res_v = nodepos

    return res_v

def get_edge_collision_length(maze, node1_state, node2_state):
    if isinstance(node1_state, list):
        node1_pos = np.array(node1_state)
    else:
        node1_pos = state_to_numpy(node1_state)
    if isinstance(node2_state, list):
        node2_pos = np.array(node2_state)
    else:
        node2_pos = state_to_numpy(node2_state)

    diff = node2_pos - node1_pos
    edge_discretization = int(np.max(np.abs(diff)) / 0.1) + 1
    step = diff / edge_discretization
    assert(np.max(step) < 0.1 and np.min(step) > -0.1)

    col_cnt = 0
    for i in range(edge_discretization):
        nodepos = node1_pos + step * i
        if not maze.is_state_valid(nodepos.tolist()):
            col_cnt += 1

    col_length = float(col_cnt) / edge_discretization * cal_edge_length(node1_pos, node2_pos)

    return col_length

def visualize_nodes(occ_g, curr_node_posns, start_pos, goal_pos, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10,10), dpi=100)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size / 4.0 - 0.25
    s = (10.0 / occ_grid_size * 100 / 2) ** 2 + 500
    for i in range(occ_grid_size):
        for j in range(occ_grid_size):
            if occ_g[i,j] == 1:
                # ax1.add_patch(patches.Rectangle(
                #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
                #     0.1,          # width
                #     0.1,          # height
                #     alpha=0.6
                #     ))
                plt.scatter(j/2.0 - tmp, tmp - i/2.0, color="black", marker='s', s=s, alpha=1) # init

    curr_node_posns = np.array(curr_node_posns)
    if len(curr_node_posns)>0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(curr_node_posns):
            visualize_robot(pos)
            plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(start_pos, start=True)
    if goal_pos is not None:
        # plt.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, edgecolors='black', alpha=1, zorder=10) # goal
        visualize_robot(goal_pos, goal=True)

    plt.title("Visualization")
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    if show:
        plt.show()
    if save:
        plt.savefig(file_name, dpi=fig1.dpi)
    plt.close()

def visualize_robot(robot_state, start=False, goal=False):
    base_x, base_y = robot_state[:2]

    if start:
        plt.gca().add_patch(patches.Rectangle((base_x-0.1, base_y-0.1), 0.2, 0.2, facecolor='y'))
    elif goal:
        plt.gca().add_patch(patches.Rectangle((base_x-0.1, base_y-0.1), 0.2, 0.2, facecolor='r'))
    else:
        plt.gca().add_patch(patches.Rectangle((base_x-0.1, base_y-0.1), 0.2, 0.2, facecolor='g'))