from tqdm import tqdm
import numpy as np
import random

from dataset import *
from utils.logging import Logging
import torch
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.djsccf import *

class DJSCCFTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = DJSCCF_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss(reduction='mean')
        self.base_snr = args.base_snr
    def train(self):
        for epoch in range(self.args.out_e):
            print(f"Epoch {epoch}")
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            self.model.train()
            for x, y in tqdm(self.train_dl):
                x, y = x.to(self.device), y.to(self.device)
                recon_x, mu, logvar = self.model(x)
                loss = self.model.loss(recon_x, x, mu, logvar)
                #loss = self.criterion(recon_x, x) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.detach().item()
            epoch_train_loss /= (len(self.train_dl)) 
            print('Epoch Loss:', epoch_train_loss)
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)

            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    test_rec, test_mu, test_logvar = self.model(test_imgs)
                    loss = self.model.loss( test_rec, test_imgs, test_mu, test_logvar)
                    #loss = self.criterion(test_rec, test_imgs)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= (len(self.test_dl))
                print('Validation Loss:', epoch_val_loss)
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)

            # Saving checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()
        
