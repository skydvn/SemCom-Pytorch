from tqdm import tqdm
import numpy as np

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
        
class DJSCCNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.rec_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, rec, img):
        rec_loss = self.rec_loss(rec, img)

        cls_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        inv_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        var_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        irep_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        kld_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)

        psnr_val = 20 * torch.log10(torch.max(img) / torch.sqrt(rec_loss))

        total_loss = args.rec_coeff * rec_loss

        return {
            "cls_loss": cls_loss,
            "rec_loss": rec_loss,
            "psnr_loss": psnr_val,
            "kld_loss": kld_loss,
            "inv_loss": inv_loss,
            "var_loss": var_loss,
            "irep_loss": irep_loss,
            "total_loss": total_loss
        }