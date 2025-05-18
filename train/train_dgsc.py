from tqdm import tqdm
import numpy as np
import os
import glob

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.dgsc import *


class DGSCTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.model = DGSC_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = DGSCLoss()
        self.domain_list = ['AWGN', 'Rayleigh', 'Rician']

    def train(self):
        domain_list = self.domain_list
        rec = [ [] for _ in domain_list ]
        for epoch in range(self.args.out_e):
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            self.model.train()
            for batch_idx, (x, y) in enumerate(tqdm(self.train_dl)):
                x, y = x.to(self.device), y.to(self.device)
                channel_losses = []  # loss của từng kênh
                total_loss = 0  # Tổng loss để backward 
                for i, domain_str in enumerate(domain_list):
                    # TODO: Loop over domain_list
                    # TODO: Infer x through model with different settings
                    # TODO: x --Encoder--> z --Channel--> z' --Decoder--> x'
                    out = self.model.channel_perturb(x, domain_str)
                    channel_loss = self.criterion.forward(self.args, out, x)
                    channel_losses.append(channel_loss.item())  # Lưu loss của từng kênh
                    total_loss += channel_loss 
                    if batch_idx % 50 == 0: 
                        tqdm.write(f'Lap {i}, domain: {domain_str}, SNR: {self.args.base_snr}, epoch: {epoch}, loss: {channel_loss.item()}')
                    #rec[i].append(out)
                    #print(f'Channel Loss of {domain_str}: {channel_loss.item()}')  # In loss của từng kênh
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            
                epoch_train_loss += total_loss.detach().item()
                #print(f'Channel Losses: {channel_losses}')  # In loss của từng kênh 

            epoch_train_loss /= len(self.train_dl)
            print('Epoch Loss:', epoch_train_loss)
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)

            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    test_rec = self.model(test_imgs)
                    loss = self.criterion.forward(self.args, test_imgs, test_rec)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= len(self.test_dl)
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                print('Validation Loss:', epoch_val_loss)
            # Lưu checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()
     # def domain_gen(self, x, domain_list):
    #     """
    #     :param x: input images
    #     :param domain_list: list of settings
    #     :return:
    #     [x1, x2, x3, ... x4]: the generated noise with different channels from x
    #     """
    #     # TODO: Loop over domain_list (for example: [AWGN10, Rayleigh15]
    #
    #     return
        
class DGSCLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.rec_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, rec, img):
        rec_loss = self.rec_loss(rec, img)
        total_loss = args.rec_coeff * rec_loss

        return total_loss


