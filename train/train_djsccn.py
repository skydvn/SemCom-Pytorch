from tqdm import tqdm
import numpy as np

from dataset import *
from utils.logging import Logging
import torch
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.djsccn import *

class DJSCCNTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = DJSCCN_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = DJSCCNLoss()
        
    def train(self):

        for epoch in range(self.args.out_e):
            psnr_train = 0

            self.model.train()
            for x, y in tqdm(self.train_dl):
                x, y = x.to(self.device), y.to(self.device)
                rec = self.model(x)
                loss_dict = self.criterion(self.args, x, rec)
                
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                self.optimizer.step()
                
                for key, value in loss_dict.items():
                    self.log_interface(key=f"train/loss/{key}", value=value.item())

                psnr_train += loss_dict['psnr_loss'].item()

            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    test_rec = self.model(test_imgs)
                    loss_dict = self.criterion(self.args, test_imgs, test_rec)
                    
            for key, value in loss_dict.items():
                self.log_interface(key=f"test/loss/{key}", value=value.item())
            
            self.log_interface.step(epoch=epoch, test_len=len(self.test_dl))
        
