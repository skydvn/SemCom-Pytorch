import os
import pandas
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm
import numpy as np

import torch
from torch import nn


class NECST_CIFAR(nn.Module):
    def __init__(self, args, in_channel, class_num):
        super(NECST_CIFAR, self).__init__()

        self.in_channel = in_channel
        self.class_num = class_num
        self.inv_cdim = args.inv_cdim    # int(32)  # inv_cdim
        self.var_cdim = args.var_cdim    # int(32)  # var_cdim
        self.ib_cdim = self.inv_cdim + self.var_cdim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=self.var_cdim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.var_cdim),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.ib_cdim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        rec = self.decoder(enc)
        return rec

    def get_latent(self, x):
        enc = self.encoder(x)
        return enc

    def get_semcom_recon(self, x, n_var, ukie_flag, device):
        enc = self.encoder(x)

        # generate Gaussian propagating noise
        noise = torch.normal(mean=torch.zeros(enc.size()),
                             std=torch.ones(enc.size()) * n_var).to(device)
        print(n_var)
        # feedforward latents
        if not ukie_flag:
            noise_i = torch.normal(mean=torch.zeros(enc.size()),
                                   std=torch.ones(enc.size()) * n_var).to(device)
            inv = enc + noise_i
        else:
            inv = enc
        var = enc + noise  # simulate a noise by physical channels

        rec = self.decoder(var)
        return rec

    def get_latent_size(self, x):
        enc = self.encoder(x)
        return enc.size()