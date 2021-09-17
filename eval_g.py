import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import networkx as nx
import random
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import math
import argparse

import lego.utils as utils
from env.maze_2d import Maze2D

from model_g import MyModel

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='3')
parser.add_argument('--dist', action='store_true', default=False)
args = parser.parse_args()

# constants
dim = 4
occ_grid_dim = 100
path = osp.join(CUR_DIR, "models/model_{}_g.pt".format(args.name))

# hyperparameters
bs = 32
lr = 1e-3
num_steps = 10000
alpha = 0.5

def contrastive_loss(mu, sigma, samples, num_pos_samples, num_neg_samples, device='cpu'):
    bs = mu.shape[0]
    total_loss = 0
    pos_llh_sum = 0
    for i in range(bs):
        pos_samples = samples[i, :num_pos_samples[i]]
        neg_samples = samples[i, num_pos_samples[i]:num_pos_samples[i]+num_neg_samples[i]]
        cov = torch.eye(mu[i].shape[0], device=device) * sigma[i]
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i], cov)
        pos_llh = torch.exp(m.log_prob(pos_samples))
        neg_llh = torch.exp(m.log_prob(neg_samples))
        # pos_llh = torch.exp(-0.5 * torch.pow((pos_samples - mu) / sigma, 2)) / (sigma * 2.50662827463) # sqrt(2pi)
        # neg_llh = torch.exp(-0.5 * torch.pow((neg_samples - mu) / sigma, 2)) / (sigma * 2.50662827463) # sqrt(2pi)
        # print("pos:", torch.sum(pos_llh))
        # print("neg", torch.mean(neg_llh))
        # total_loss += alpha1 * torch.mean(pos_llh) + alpha2 * (torch.exp(torch.mean(pos_llh)) - torch.exp(torch.mean(neg_llh)))
        pos_llh_sum += torch.sum(pos_llh)
        loss = torch.log(torch.sum(pos_llh) / (torch.sum(pos_llh) + torch.sum(neg_llh)))
        has_nan = torch.isnan(loss).any()
        assert not has_nan
        total_loss += loss
    loss = -(total_loss / bs)
    # loss = mse_loss_val
    return loss, (pos_llh_sum / bs)

def pos_loss(mu, samples, num_pos_samples, device='cpu'):
    bs = mu.shape[0]
    mse_loss = torch.nn.MSELoss()
    total_pos_samples = torch.zeros((bs, dim)).to(device)
    for i in range(bs):
        pos_samples = samples[i, :num_pos_samples[i]]
        total_pos_samples[i] = pos_samples
    loss = mse_loss(mu, total_pos_samples)
    return loss

def dist_loss(pred_dist, tgt_dist, device='cpu'):
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(pred_dist, tgt_dist)
    return loss

class MyDataset(Dataset):
    def __init__(self, dir, transform=None, target_transform=None, device="cpu"):
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        self.dataset = self.load_dataset_from_file()

        print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start_pos, goal_pos, occ_grid, next_pos, non_connectable_nodes, dist = self.dataset[idx]

        dim = len(start_pos)
        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        dist = torch.Tensor([dist])
        occ_grid = torch.Tensor(occ_grid).view(1, 10, 10)
        samples = torch.zeros((500, dim))
        num_pos_samples = 1
        num_neg_samples = len(non_connectable_nodes)
        samples[:num_pos_samples] = torch.Tensor(next_pos)
        samples[num_pos_samples:num_pos_samples + num_neg_samples] = torch.Tensor(non_connectable_nodes)
        # pos_samples = torch.Tensor(pos_samples)
        # neg_samples = torch.Tensor(neg_samples)

        # if self.transform:
        #     state = self.transform(state)
        #     occ_grid = self.transform(occ_grid)
        # if self.target_transform:
        #     pos_samples = self.target_transform(pos_samples)
        #     neg_samples = self.target_transform(neg_samples)

        input = torch.cat((start_pos, goal_pos), dim=0).to(self.device)
        start_pos = start_pos.to(self.device)
        goal_pos = goal_pos.to(self.device)
        occ_grid = occ_grid.to(self.device)
        # output = torch.cat((pos_samples, neg_samples), dim=0).to(self.device)
        # pos_samples = pos_samples.to(self.device)
        # neg_samples = neg_samples.to(self.device)
        samples = samples.to(self.device)
        dist = dist.to(self.device)
        num_pos_samples = torch.as_tensor(num_pos_samples).to(self.device)
        num_neg_samples = torch.as_tensor(num_neg_samples).to(self.device)
        return input, occ_grid, samples, num_pos_samples, num_neg_samples, dist

    def load_dataset_from_file(self):
        file_path = osp.join(CUR_DIR, "dataset/data_{}_g.json".format(args.name))
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        return dataset

def visualize_robot(robot_state, start=False, goal=False, mu=False):
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
    elif mu:
        plt.plot((base_x, x1), (base_y, y1), 'co-')
        plt.plot((x1, x2), (y1, y2), 'bo-')
    else:
        plt.plot((base_x, x1), (base_y, y1), 'go-')
        plt.plot((x1, x2), (y1, y2), 'bo-')

def visualize_nodes(occ_g, curr_node_posns, mu, sigma, start_pos, goal_pos, show=True, save=False, file_name=None):
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

    if len(curr_node_posns)>0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(curr_node_posns):
            visualize_robot(pos)
            plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(start_pos, start=True)
    if goal_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(goal_pos, goal=True)
    if mu is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(mu, mu=True)
    if sigma is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        base_x, base_y = mu[:2]
        plt.plot((base_x, base_x + sigma[0]), (base_y, base_y), 'g-')
        plt.plot((base_x, base_x), (base_y, base_y + sigma[1]), 'g-')

    plt.title("Visualization")
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    if show:
        plt.show()
    if save:
        plt.savefig(file_name, dpi=fig1.dpi)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Initialize the MLP
in_dim = dim * 2
out_dim = dim * 2
mlp = MyModel(in_dim, out_dim).to(device)

# Define the loss function and optimizer
# optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1000, verbose=True, factor=0.5)

# dataset and dataloader
dataset = MyDataset(None, None, device=device)
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
print("train size: {}. test_size: {}".format(train_size, test_size))
train_set = torch.utils.data.Subset(dataset, np.arange(train_size))
val_set = torch.utils.data.Subset(dataset, np.arange(train_size, len(dataset)))
train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(val_set, batch_size=1, shuffle=True)

print("loading weights from {}".format(path))
mlp.load_state_dict(torch.load(path))
mlp.eval()

print("Evaluting on eval dataset!!!")
total_loss = 0
total_pos_llh = 0
total_pos_loss = 0
total_dist_loss = 0
total_contra_loss = 0
eval_iter = iter(eval_dataloader)
for i in range(len(eval_dataloader)):
    input, occ_grid, samples, num_pos_samples, num_neg_samples, tgt_dist = next(eval_iter)

    mu, sigma, dist = mlp(input, occ_grid)

    if i < 10:
        cov = torch.eye(mu[0].shape[0], device=device) * sigma[0]
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[0], cov)
        # samples = m.sample_n(10).cpu().numpy()
        sample = mu[0].detach().cpu().numpy().reshape(1, -1)
        input = input.detach().cpu().numpy().reshape(dim * 2)
        start = input[:dim]
        goal = input[dim:]
        occ_grid = occ_grid.detach().cpu().numpy().reshape(10, 10)
        myu = mu[0].detach().cpu().numpy()
        sigma = sigma[0].detach().cpu().numpy()

        mse_loss = torch.nn.MSELoss()
        pos_samples = samples[0, :num_pos_samples[0]]
        mse_loss_val = mse_loss(mu, pos_samples)
        tmp = pos_samples[0].detach().cpu().numpy().reshape(-1)
        print(tmp)
        print(myu)
        # print(mse_loss_val)
        print(start)
        print(goal)
        visualize_nodes(occ_grid, [tmp], myu, sigma, start, goal)

    # Compute loss
    pos_loss_val = pos_loss(mu, samples, num_pos_samples, device=device)
    loss = pos_loss_val
    if args.dist:
        dist_loss_val = dist_loss(dist, tgt_dist)
        loss = loss + dist_loss_val
        total_dist_loss += dist_loss_val.item()
    contra_loss_val, pos_llh_sum = contrastive_loss(mu, sigma, samples, num_pos_samples, num_neg_samples, device=device)
    loss = loss + contra_loss_val

    total_loss += loss.item()
    total_pos_llh += pos_llh_sum.item()
    total_pos_loss += pos_loss_val.item()
    total_contra_loss += contra_loss_val.item()

print("Evaluation----")
print('-----------------Total loss : %.3f' % (total_loss / len(eval_dataloader)))
print('contra_loss: %.3f' % (total_contra_loss / len(eval_dataloader)))
print('pos_loss : %.3f' % (total_pos_loss / len(eval_dataloader)))
print('pos llh : %.3f' % (total_pos_llh / len(eval_dataloader)))

print("Evaluting on maze!!!")
maze = Maze2D()
for _ in range(10):
    maze.clear_obstacles()
    maze.random_obstacles(mode=3)
    maze.sample_start_goal()
    occ_grid = np.array(maze.get_occupancy_grid()).reshape(50, 50)
    occ_grid = utils.shrink_occ_grid(occ_grid)
    occ_grid = torch.Tensor(occ_grid).view(1,1,10,10)
    cur_state = np.array(maze.start)
    goal = np.array(maze.goal)
    for _ in range(1):
        context = np.concatenate((cur_state, goal))
        context = torch.Tensor(context).view(1, -1)

        mu, sigma, dist = mlp(context, occ_grid)

        cov = torch.eye(mu[0].shape[0], device=device) * sigma[0]
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[0], cov)
        samples = m.sample_n(10).cpu().numpy()
        # samples = mu[0].detach().cpu().numpy().reshape(1, -1)
        context = context.detach().cpu().numpy().reshape(dim * 2)
        start = context[:dim]
        goal = context[dim:]
        occ_grid_viz = occ_grid.detach().cpu().numpy().reshape(10, 10)
        myu = mu[0].detach().cpu().numpy()
        sigma = sigma[0].detach().cpu().numpy()

        print(start)
        print(goal)
        print(sigma)
        visualize_nodes(occ_grid_viz, samples, myu, sigma, start, goal)

        cur_state = myu