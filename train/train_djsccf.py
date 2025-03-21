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
            epoch_train_loss = 0
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
                    loss = self.criterion.forward(self.args, x, rec)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_train_loss += loss.detach().item() # Check layer effect later

                    f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                       std=torch.ones(img[:, 0:1, :].size()) * n_var).to(self.device)

            epoch_train_loss = epoch_train_loss / len(self.train_dl)
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)

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
                        test_loss = self.criterion.forward(self.args, x, test_rec)

class DJSCCNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.rec_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, rec, img):
        rec_loss = self.rec_loss(rec, img)
        total_loss = args.rec_coeff * rec_loss

        return total_loss

