from tqdm import tqdm
import numpy as np
import os
import glob
import wandb

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.djsccn import *

class DJSCCNTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = DJSCCN_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = DJSCCNLoss()
        self.base_snr = args.base_snr
        
    def train(self):

        for epoch in range(self.args.out_e):
            print(f"Epoch {epoch}")
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            self.model.train()
            for x, y in tqdm(self.train_dl):
                x, y = x.to(self.device), y.to(self.device)
                rec = self.model(x)
                loss = self.criterion.forward(self.args, x, rec)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.detach().item()
            epoch_train_loss /= (len(self.train_dl))
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            if self.args.wandb:
                wandb.log({'train/loss': epoch_train_loss}, step=epoch)

            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    test_rec = self.model.get_train_recon(test_imgs, self.base_snr)
                    loss = self.criterion.forward(self.args, test_imgs, test_rec)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= (len(self.test_dl))
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                if self.args.wandb:
                    wandb.log({'val/loss': epoch_val_loss}, step=epoch)

            # Saving checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()
        
class DJSCCNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.rec_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, rec, img):
        rec_loss = self.rec_loss(rec, img)
        total_loss = args.rec_coeff * rec_loss

        return total_loss