import torch
import networkx as nx
import random
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import argparse
import matplotlib.pyplot as plt

from model import AE

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', default='2')
parser.add_argument('--checkpoint', default='')
parser.add_argument('--embedding_size', type=int, default=16)
args = parser.parse_args()

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
        occ_grid = self.dataset[idx]

        occ_grid = torch.Tensor(occ_grid).view(1, 10, 10)
        occ_grid = occ_grid.to(self.device)
        return occ_grid

    def load_dataset_from_file(self):
        occ_grids = []
        for i in range(10000):
            print("loading env {}".format(i))
            data_dir = osp.join(CUR_DIR, "dataset/{}/{}".format(args.name, i))
            occ_grid = np.loadtxt(osp.join(data_dir, "occ_grid.txt")).tolist()
            occ_grids.append(occ_grid)

        return occ_grids

def loss_function(recon_x, x):
    bs_size = recon_x.shape[0]
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(recon_x.view(bs_size, -1), x.view(bs_size, -1))
    return loss

def visualize_occ_grid(occ_g, recon_occ_g, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(121, aspect='equal')
    ax2 = fig1.add_subplot(122, aspect='equal')
    # for i in range(10):
    #     for j in range(10):
    #         if occ_g[i,j] == 1:
    #             # ax1.add_patch(patches.Rectangle(
    #             #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
    #             #     0.1,          # width
    #             #     0.1,          # height
    #             #     alpha=0.6
    #             #     ))
    #             ax1.scatter(j/2.0 - 2.25, 2.25 - i/2.0, color="black", marker='s', s=900, alpha=1) # init
    ax1.imshow(occ_g)

    # for i in range(10):
    #     for j in range(10):
    #         if recon_occ_g[i,j] == 1:
    #             # ax1.add_patch(patches.Rectangle(
    #             #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
    #             #     0.1,          # width
    #             #     0.1,          # height
    #             #     alpha=0.6
    #             #     ))
    #             ax2.scatter(j/2.0 - 2.25, 2.25 - i/2.0, color="black", marker='s', s=1000, alpha=1) # init
    ax2.imshow(recon_occ_g)

    plt.title("Visualization")
    # ax1.set_xlim(-2.5,2.5)
    # ax1.set_ylim(-2.5,2.5)
    # ax2.set_xlim(-2.5,2.5)
    # ax2.set_ylim(-2.5,2.5)
    if show:
        plt.show()
    if save:
        plt.savefig(file_name, dpi=fig1.dpi)

bs = 32
lr = 1e-3
weight_decay = 1e-6
num_steps = 20000
num_epoch = 5
path = osp.join(CUR_DIR, 'models/model_{}.pt'.format(args.name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MyDataset(None, None, device=device)
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
print("train size: {}. test_size: {}".format(train_size, test_size))
# train_set = torch.utils.data.Subset(dataset, np.arange(train_size))
# val_set = torch.utils.data.Subset(dataset, np.arange(train_size, len(dataset)))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))
train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(val_set, batch_size=1)

model = AE(args.embedding_size)
model.to(device)
if args.checkpoint != '':
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.train()
train_loss = 0
train_iter = iter(train_dataloader)
i = 0
while i < num_steps:
    for occ_grid in train_dataloader:
        # Get inputs
        # occ_grid = next(train_iter)
        # occ_grid = occ_grid.to(device)
        optimizer.zero_grad()
        recon_batch = model(occ_grid)
        loss = loss_function(recon_batch, occ_grid)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        i +=1

        # Print statistics
        current_loss = loss.item()
        if i % 100 == 0:
            print('Loss after mini-batch %5d: %.3f' % (i, current_loss))
        if i % 500 == 0:
            torch.save(model.state_dict(), path)
            print("saved session to ", path)

# eval
model.load_state_dict(torch.load(path))
model.eval()
eval_iter = iter(eval_dataloader)
total_loss = 0
for i in range(len(val_set)):
    occ_grid = next(eval_iter)
    occ_grid = occ_grid.to(device)
    recon_grid = model(occ_grid)
    loss = loss_function(recon_grid, occ_grid)
    total_loss += loss

    occ_grid = occ_grid[0].detach().cpu().numpy().reshape(10, 10)
    recon_grid = recon_grid[0].detach().cpu().numpy().reshape(10, 10)
    # recon_grid = np.rint(recon_grid[0].detach().cpu().numpy().reshape(10, 10))

    if i < 10:
        visualize_occ_grid(occ_grid, recon_grid)
total_loss /= len(val_set)
print(total_loss)