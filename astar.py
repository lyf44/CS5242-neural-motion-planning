import heapq
from math import sqrt
import numpy as np

import utils

def get_successors(G, cur):
    return G.neighbors(cur)

def compute_distance(start_config, end_config):
    s = [(start_config[i] - end_config[i])**2 for i in range(0, len(start_config))]
    return sqrt(sum(s))

def compute_distance_id(G, u, v):
    start_config = utils.state_to_numpy(G.nodes[u]['coords'])
    end_config = utils.state_to_numpy(G.nodes[v]['coords'])
    return compute_distance(start_config, end_config)

def get_heuristic(G, v, goal_v):
    return compute_distance_id(G, v, goal_v)

def astar(G, start_v, goal_v, occ_g, row, col, inc, h_weight=1):
    queue = []
    heapq.heappush(queue, (0, 0, start_v))
    nodes = dict()
    nodes[start_v] = (0, [])

    count = 0
    while len(queue):
        heu, dis, cur = heapq.heappop(queue)

        if dis > nodes[cur][0]:
            continue

        if cur == goal_v:
            addv = goal_v
            plan = []
            while addv != []:
                plan.append(addv)
                addv = nodes[addv][1]
            # print(" count = ", count)
            # print(" dis = ", dis)
            return np.array(plan[::-1]), dis

        next_cur = get_successors(G, cur)

        for v in next_cur:
            # dis_v = dis + compute_distance_id(G, cur, v)
            dis_v = dis + G[cur][v]['weight']
            # dis_v = dis + helper.calc_weight_states(G.nodes[cur]['coords'], G.nodes[v]['coords'])

            if (v not in nodes) or nodes[v][0] > dis_v:
                count += 1
                cost_v =  dis_v + h_weight * get_heuristic(G, v, goal_v)
                node1_pos = utils.state_to_numpy(G.nodes[v]['coords'])
                node2_pos = utils.state_to_numpy(G.nodes[cur]['coords'])

                lines = []
                colors = []
                lines.append([node1_pos, node2_pos])
                # if not helper.is_edge_free(node1_pos, node2_pos, occ_g, row, col, inc = inc):
                #     colors.append((1,0,0,0.3))
                #     lc = mc.LineCollection(lines, colors=colors, linewidths=1)
                #     continue
                colors.append((0,1,0,0.3))
                heapq.heappush(queue, (cost_v, dis_v, v))
                nodes[v] = (dis_v, cur)
    # print(" count = ", count)
    return [], None