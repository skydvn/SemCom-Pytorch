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
        self.criterion = DJSCCNLoss()

    def train(self):

        for epoch in range(self.args.out_e):
            psnr_train = 0
            n_var = random.random() * 0.1

            self.model.train()
            for img, _ in tqdm(self.train_dl):
                f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                   std=torch.ones(img[:, 0:1, :].size()) * 0).to(self.device)
                for layer in range(3):
                    img_c = img[:, layer:layer+1, :].to(self.device)
                    x = torch.cat((img_c, img_c + f_noise), dim=1)
                    rec = self.model(x)
                    loss_dict = self.criterion.forward(self.args, x, rec)
                    
                    self.optimizer.zero_grad()
                    loss_dict["total_loss"].backward()
                    self.optimizer.step()
                    
                    f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                       std=torch.ones(img[:, 0:1, :].size()) * n_var).to(self.device)

                    for key, value in loss_dict.items():
                        self.log_interface(key=f"train/loss/{key}", value=value.item())

                    psnr_train += loss_dict['psnr_loss'].item()

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total, rec_val, kld_val, inv_val, var_val, cls_val, psnr_val = 0, 0, 0, 0, 0, 0, 0
                for test_imgs, test_labels in tqdm(self.test_dl):
                    f_noise = torch.normal(mean=torch.zeros(test_imgs[:, 0:1, :].size()),
                                        std=torch.ones(test_imgs[:, 0:1, :].size()) * 0).to(self.device)
                    for layer in range(3):
                        img_c = test_imgs[:, layer:layer+1, :].to(self.device, non_blocking=True)
                        x = torch.cat((img_c, img_c + f_noise), dim=1)
                        test_rec = self.model(x)
                        test_loss_dict = self.criterion.forward(self.args, x, test_rec)

                        total += test_loss_dict["total_loss"].item()
                        rec_val += test_loss_dict["rec_loss"].item()
                        kld_val += test_loss_dict["kld_loss"].item()
                        inv_val += test_loss_dict["inv_loss"].item()
                        var_val += test_loss_dict["var_loss"].item()
                        cls_val += test_loss_dict["cls_loss"].item()
                        psnr_val += test_loss_dict["psnr_loss"].item()

                total = total / len(self.test_dl)
                rec_val = rec_val / len(self.test_dl)
                kld_val = kld_val / len(self.test_dl)
                inv_val = inv_val / len(self.test_dl)
                var_val = var_val / len(self.test_dl)
                cls_val = cls_val / len(self.test_dl)
                psnr_val = psnr_val / len(self.test_dl)

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

