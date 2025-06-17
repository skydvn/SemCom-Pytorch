# from tqdm import tqdm
# import numpy as np
# import os
# import glob
# import time

# import torch
# from torch import nn
# from torch.optim import Adam

# from train.train_base import BaseTrainer
# from models.swinjscc import *
# from modules.distortion import Distortion

# import torch
# from torch.optim import Adam
# from tqdm import tqdm

# from models.swinjscc import SWINJSCC
# #from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

# class SWINJSCCTrainer(BaseTrainer):
#     def __init__(self, args):
#         super().__init__(args)
#         # Khởi tạo model SwinJSCC với args
#         self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
#         self.optimizer = Adam(self.model.parameters(), lr=args.lr)
#         self.criterion = nn.MSELoss(reduction='mean')
        
#         self.domain_list = args.domain_list

#     def parse_domain(self, domain_str):
#         """Extract channel name and SNR from domain string."""
#         channel_name = ''.join([c for c in domain_str if not c.isdigit()])
#         snr = ''.join([c for c in domain_str if c.isdigit()])
#         return channel_name, int(snr)
    

#     def train(self):
#         domain_list = self.domain_list 
#         for epoch in range(self.args.out_e):
#             self.model.train()

#             epoch_loss = 0
#             epoch_val_loss = 0
#             for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
#                 x, y = x.to(self.device), y.to(self.device)
#                 total_loss = 0  # Tổng loss để backward 
#                 channel_losses = []
#                 for i, domain_str in enumerate(domain_list):
#                     channel_type, snr = self.parse_domain(domain_str)
#                     out = self.model.channel_perturb(x, channel_type, snr)
#                     channel_loss = self.criterion(x, out)
#                     channel_losses.append(channel_loss.item())  # Lưu loss của từng kênh
#                     total_loss += channel_loss 
#                     if batch_idx % 50 == 0: 
#                         tqdm.write(f'Domain: {channel_type}, SNR: {snr}, epoch: {epoch}, loss: {channel_loss.item()}')

#                 # Backward
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()

#                 epoch_loss += total_loss.item()

#             avg_loss = epoch_loss / len(self.train_dl)
#             self.writer.add_scalar('train/loss', avg_loss, epoch)
#             print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

#             # Validation
#             self.model.eval()
#             with torch.no_grad():
#                 for test_imgs, test_labels in tqdm(self.test_dl):
#                     test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
            
#                     test_rec,_,_ = self.model.forward(test_imgs,snr)
#                     loss = self.criterion(test_imgs, test_rec)
#                     epoch_val_loss += loss.detach().item()
#                 epoch_val_loss /= len(self.test_dl)
#                 self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
#                 print('Validation Loss:', epoch_val_loss)
#             # Lưu checkpoint
#             self.save_model(epoch=epoch, model=self.model)

#         self.writer.close()
#         self.save_config()


from tqdm import tqdm
import numpy as np
import os
import glob
import time

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.swinjscc import *
from modules.distortion import Distortion

import torch
from torch.optim import Adam
from tqdm import tqdm

from models.swinjscc import SWINJSCC
#from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

class SWINJSCCTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss(reduction='mean')
        self.base_snr = 12
        self.channel_type = args.channel_type 

    def train(self):
        for epoch in range(self.args.out_e):
            self.model.train()
            self.model.channel = Channel(channel_type=self.channel_type, snr=self.base_snr)
            total_loss = 0.0
            epoch_val_loss = 0.0
            """ Vì HR_Image không trar về tuple gồm ảnh và label mà chỉ trả về ảnh thôi"""
            for batch in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                # Forward
                recon  = self.model(images,self.base_snr)
                loss = self.criterion(images, recon)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

            # Validation
            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    self.model.change_channel(channel_type = 'Rayleigh', snr = 13)
                    test_rec= self.model.forward(test_imgs,13)
                    loss = self.criterion(test_imgs, test_rec)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= len(self.test_dl)
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                print('Validation Loss:', epoch_val_loss)
            # Lưu checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()
