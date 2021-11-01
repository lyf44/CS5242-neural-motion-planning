import torch
import torch.nn as nn
import numpy as np

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 10, 10)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=2,
                     padding=1), # out = (16, 5, 5)
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1), # out = (32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1), # out = (64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1), # out = (128, 1, 1)
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(),
            nn.Flatten()
            # nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            # nn.BatchNorm2d(self.channel_mult*16),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # self.flat_fts = self.get_flat_fts(self.conv)
        self.flat_fts = 128

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
        )

    def get_flat_fts(self, fts):
        tmp = torch.ones(1, *self.input_size)
        f = fts(tmp)
        return int(np.prod(f.size()[1:]))

    def extract_feat(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 10, 10)):
        super(CNN_Decoder, self).__init__()
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 1
        self.fc_output_dim = 128 # 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.LeakyReLU()
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4, 4, 2, 1, bias=False), # out =(64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 3, 2, 1, bias=False), # out =(32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 3, 2, 1, bias=False), # out =(16, 5, 5)
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False), # out =(1, 10, 10)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x

class AE(nn.Module):
    def __init__(self, embedding_size=64):
        super(AE, self).__init__()
        self.encoder = CNN_Encoder(embedding_size)
        self.decoder = CNN_Decoder(embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)