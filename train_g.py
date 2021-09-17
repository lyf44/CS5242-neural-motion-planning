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
parser.add_argument('--param', default='gaussian')
parser.add_argument('--dist', action='store_true', default=False)
parser.add_argument('--checkpoint', default='')
args = parser.parse_args()

# constants
dim = 4
occ_grid_dim = 100
if args.dist:
    path = osp.join(CUR_DIR, "models/model_{}_g_dist.pt".format(args.name))
else:
    path = osp.join(CUR_DIR, "models/model_{}_g.pt".format(args.name))

# hyperparameters
bs = 64
lr = 1e-4
num_steps = 50000
num_epochs = 150
pretrain_epoch = 50
alpha_pos = 1.0
alpha_dist = 1.0
alpha_contra = 1.0

def state_to_numpy(state):
    strlist = state.split(',')
    # strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

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

# Set fixed random number seed
# torch.manual_seed(42)

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
if args.param == 'gaussian':
    out_dim = dim * 2
    model = MyModel(in_dim, out_dim).to(device)
elif args.param == 'dofatt':
    exit()

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
        input, occ_grid, samples, num_pos_samples, num_neg_samples, tgt_dist = data

        # # Print epoch
        # print(f'Starting epoch {epoch+1}')

        # # Set current loss value
        # current_loss = 0.0

        # Iterate over the DataLoader for training data
        # for i, data in enumerate(train_dataloader):

        # Get inputs
        # input, occ_grid, samples, num_pos_samples, num_neg_samples = next(iter(train_dataloader))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        mu, sigma, dist = model(input, occ_grid)
        # mu, sigma = outputs[:, :dim], outputs[:, dim:]

        # Compute loss
        pos_loss_val = alpha_pos * pos_loss(mu, samples, num_pos_samples, device=device)
        loss = pos_loss_val
        if args.dist:
            dist_loss_val = alpha_dist * dist_loss(dist, tgt_dist)
            loss = loss + dist_loss_val
        if epoch > pretrain_epoch:
            contra_loss_val, pos_llh_sum = contrastive_loss(mu, sigma, samples, num_pos_samples, num_neg_samples, device=device)
            contra_loss_val *= alpha_contra
            loss = loss + contra_loss_val
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
        if epoch > pretrain_epoch:
            contra_loss = contra_loss_val.item()
            pos_llh_sum = pos_llh_sum.item()
        else:
            contra_loss = 0
            pos_llh_sum = 0
        if i % 100 == 0:
            print('-----------------Total loss after mini-batch %5d, epoch %d : %.3f' % (i, epoch, current_loss))
            print('constrastive loss after mini-batch %5d, epoch %d : %.3f' % (i, epoch, contra_loss))
            print('pos_loss after mini-batch %5d, epoch %d: %.3f' % (i, epoch, pos_loss_val.item()))
            if args.dist:
                print('dist_loss after mini-batch %5d, epoch %d: %.3f' % (i, epoch, dist_loss_val.item()))

            print('pos llh after mini-batch %5d, epoch %d: %.3f' % (i, epoch, pos_llh_sum))
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
            input, occ_grid, samples, num_pos_samples, num_neg_samples, tgt_dist = data
            # Perform forward pass
            mu, sigma, dist = model(input, occ_grid)
            # mu, sigma = outputs[:, :dim], outputs[:, dim:]

            # Compute loss
            pos_loss_val = alpha_pos * pos_loss(mu, samples, num_pos_samples, device=device)
            loss = pos_loss_val
            if args.dist:
                dist_loss_val = alpha_dist * dist_loss(dist, tgt_dist)
                loss = loss + dist_loss_val
                total_dist_loss += dist_loss_val.item()
            contra_loss_val, pos_llh_sum = contrastive_loss(mu, sigma, samples, num_pos_samples, num_neg_samples, device=device)
            contra_loss_val *= alpha_contra
            loss = loss + contra_loss_val

            total_loss += loss.item()
            total_pos_llh += pos_llh_sum.item()
            total_pos_loss += pos_loss_val.item()
            total_contra_loss += contra_loss_val.item()

        print("Evaluation----")
        print('-----------------Total loss after epoch %5d: %.3f' % (epoch, total_loss / len(eval_dataloader)))
        print('contra_loss after epoch %5d: %.3f' % (epoch, total_contra_loss / len(eval_dataloader)))
        print('pos_loss after epoch %5d: %.3f' % (epoch, total_pos_loss / len(eval_dataloader)))
        if args.dist:
            print('dist_loss after epoch %5d: %.3f' % (epoch, total_dist_loss / len(eval_dataloader)))

        print('pos llh after epoch %5d: %.3f' % (epoch, total_pos_llh / len(eval_dataloader)))

# Process is complete.
print('Training process has finished.')