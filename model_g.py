import torch
import torch.nn as nn
import os.path as osp

CUR_DIR = osp.dirname(osp.abspath(__file__))

class MyModel(nn.Module):
    def __init__(self, state_dim, out_dim):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # (bs, 64, 1, 1)
            nn.Flatten()
        )

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.param_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        self.dist_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.softplus = nn.Softplus()

    def forward(self, state, occ_grid):
        f = self.feature_extractor(occ_grid)
        x1 = torch.cat((state, f), dim=1)
        h = self.layers(x1)

        # param head
        y  = self.param_head(h)
        mu = y[:, :self.out_dim // 2]
        sigma = torch.exp(y[:, self.out_dim // 2:] * 0.5)

        # dist head
        dist = self.dist_head(h)
        return mu, sigma, dist
