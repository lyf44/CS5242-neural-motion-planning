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
import argparse

from model_g import MyModel

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='3')
parser.add_argument('--checkpoint', default='')
args = parser.parse_args()

# constants
dim = 4
occ_grid_dim = 100
path = osp.join(CUR_DIR, "models/model_{}.pt".format(args.name))

# hyperparameters
bs = 64
lr = 1e-4
num_steps = 50000
num_epochs = 150
pretrain_epoch = 50
alpha_pos = 1.0
alpha_dist = 1.0
alpha_contra = 1.0

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
        print("Loading data from {}".format(file_path))
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        return dataset

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the MLP
in_dim = dim * 2
model = MyModel(in_dim).to(device)

if args.checkpoint != '':
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))

# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1000, verbose=True, factor=0.5)

# dataset and dataloader
dataset = MyDataset(None, None, device=device)
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
print("train size: {}. test_size: {}".format(train_size, test_size))
train_set = torch.utils.data.Subset(dataset, np.arange(train_size))
val_set = torch.utils.data.Subset(dataset, np.arange(train_size, len(dataset)))
train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(val_set, batch_size=1)

# Run the training loop
i = 0
for epoch in range(num_epochs):
    model.train()
    for data in train_dataloader:
        # Get batch of data
        input, occ_grid, samples, num_pos_samples, num_neg_samples, tgt_dist = data

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass

        # Compute loss

        # print(loss)
        assert not torch.isnan(loss).any()

        # Perform backward pass
        loss.backward()

        # for name, param in model.named_parameters():
        #     print(name, param.grad.norm())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # Perform optimization
        optimizer.step()

        # scheduler.step(loss)

        # Print statistics
        current_loss = loss.item()

        if i % 100 == 0:
            print('-----------------Total loss after mini-batch %5d, epoch %d : %.3f' % (i, epoch, current_loss))
            current_loss = 0.0

        if i % 500 == 0:
            torch.save(model.state_dict(), path)
            print("saved session to ", path)

        i+=1

    if epoch % 5 == 0:
        # eval
        model.eval()
        total_loss = 0
        total_pos_llh = 0
        total_pos_loss = 0
        total_dist_loss = 0
        total_contra_loss = 0
        for data in eval_dataloader:
            # Get batch of data
            input, occ_grid, samples, num_pos_samples, num_neg_samples, tgt_dist = data

            # Perform forward pass

            # Compute loss

        print("Evaluation----")
        print('-----------------Total loss after epoch %5d: %.3f' % (epoch, total_loss / len(eval_dataloader)))
        print('contra_loss after epoch %5d: %.3f' % (epoch, total_contra_loss / len(eval_dataloader)))
        print('pos_loss after epoch %5d: %.3f' % (epoch, total_pos_loss / len(eval_dataloader)))
        if args.dist:
            print('dist_loss after epoch %5d: %.3f' % (epoch, total_dist_loss / len(eval_dataloader)))

        print('pos llh after epoch %5d: %.3f' % (epoch, total_pos_llh / len(eval_dataloader)))

# Process is complete.
print('Training process has finished.')