import numpy as np
import math
import matplotlib.pyplot as plt

def state_to_numpy(state):
    strlist = state.split(',')
    # strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

def calc_weight_states(s1, s2):
    if isinstance(s1, list):
        config1 = np.array(s1)
    else:
        config1 = state_to_numpy(s1)
    if isinstance(s2, list):
        config2 = np.array(s2)
    else:
        config2 = state_to_numpy(s2)

    # config1 = state_to_numpy(s1)
    # config2 = state_to_numpy(s2)
    return math.sqrt(float(np.sum((config2-config1)**2)))

def is_edge_free(maze, node1_state, node2_state, EDGE_DISCRETIZATION = 20, inc = 0.035):
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
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            return False
    return True

def visualize_nodes(occ_g, curr_node_posns, start_pos, goal_pos, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    for i in range(10):
        for j in range(10):
            if occ_g[i,j] == 1:
                # ax1.add_patch(patches.Rectangle(
                #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
                #     0.1,          # width
                #     0.1,          # height
                #     alpha=0.6
                #     ))
                plt.scatter(j/2.0 - 2.25, 2.25 - i/2.0, color="black", marker='s', s=1000, alpha=1) # init

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
    angle1 = robot_state[2]
    angle2 = robot_state[3]

    x1 = base_x + math.sin(angle1) * 0.5
    y1 = base_y + math.cos(angle1) * 0.5

    x2 = x1 + math.sin(angle2 + angle1) * 0.5
    y2 = y1 + math.cos(angle2 + angle1) * 0.5

    if start:
        plt.plot((base_x, x1), (base_y, y1), 'yo-')
        plt.plot((x1, x2), (y1, y2), 'bo-')
    elif goal:
        plt.plot((base_x, x1), (base_y, y1), 'ro-')
        plt.plot((x1, x2), (y1, y2), 'bo-')
    else:
        plt.plot((base_x, x1), (base_y, y1), 'go-')
        plt.plot((x1, x2), (y1, y2), 'bo-')