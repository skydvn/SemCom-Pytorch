from tqdm import tqdm
import numpy as np
import os
import glob
import time

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.swinjscc_fishr import *
from modules.distortion import Distortion

import torch
from torch.optim import Adam
from tqdm import tqdm

from models.swinjscc_fishr import SWINJSCC_FISHR
from collections import OrderedDict

from backpack import backpack, extend
from backpack.extensions import BatchGrad




class SWINJSCC_FISHRTrainer(BaseTrainer):
    
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = SWINJSCC_FISHR(args, self.in_channel, self.class_num).to(self.device)
       

        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss(reduction='mean')  
        self.num_domains = 3
        self.penalty_weight = 0.1      
        self.ema_decay      = 0.9         
        self.penalty_anneal_iters  = 5
        # #extend_all(self.model.decoder)
        # extend_all(self.model.encoder)
        # #self.model = extend(self.model)
        # self.ema_per_domain = [
        #     MovingAverage(ema=self.ema_decay, oneminusema_correction=True)
        #     for _ in range(self.num_domains)
        # ]
        # for name, m in self.model.encoder.named_modules():
        #     if isinstance(m, nn.Linear):
        #         print(f"  {name}: {m}")    
        # self.bce_extended = extend(nn.MSELoss(reduction='none'))
        self.update_count = 0
        self.domain_list = args.domain_list
        print(self.domain_list)
    def parse_domain(self, domain_str):
        """Extract channel name and SNR from domain string."""
        channel_name = ''.join([c for c in domain_str if not c.isdigit()])
        snr = ''.join([c for c in domain_str if c.isdigit()])
        return channel_name, int(snr)
    
    def train(self):
        domain_list = self.domain_list 
        for epoch in range(self.args.out_e):
            self.model.train()

            epoch_loss = 0
            epoch_val_loss = 0
            total_loss = 0
            for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
                x, y = x.to(self.device), y.to(self.device)
                
                all_in = []
                all_out = []
                len_minibatches = []
                for i, domain_str in enumerate(domain_list):
                    channel_type, snr = self.parse_domain(domain_str)
                    out = self.model.channel_perturb(x, channel_type, snr)
                    # FIXME the tensors should be flattened later
                    all_in.append(x)  
                    all_out.append(out)
                    len_minibatches.append(x.shape[0])

                all_in = torch.cat(all_in, dim=0)
                all_out = torch.cat(all_out, dim=0)
                penalty = self.compute_fishr_penalty(all_out, all_in, len_minibatches)
                loss = self.criterion(all_out, all_in) # la so thuc nen phai dung mse khong dung cross entropy 
                penalty_weight = 0.1 
                if self.update_count >= self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight 
                if self.update_count < self.penalty_anneal_iters:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                    penalty_weight = 0
                self.update_count += 1

                objective = loss + penalty_weight * penalty
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()

                #return {'loss': objective.item(), 'nll': loss.item(), 'penalty': penalty.item()}
                # Backward
                total_loss += objective.item()
            avg_loss = total_loss / len(self.train_dl)
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

            # Validation
            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    self.model.change_channel(channel_type = 'Rayleigh', snr = 13)
                    test_rec = self.model.forward(test_imgs,snr)
                    loss = self.criterion(test_imgs, test_rec)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= len(self.test_dl)
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                print('Validation Loss:', epoch_val_loss)
            # Lưu checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()

    def l2_between_dicts(dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

    def compute_fishr_penalty(self, all_out, all_in, len_minibatches):
        grads = []
        idx = 0

    # 1) Chia theo từng domain
        for bsize in len_minibatches:
            rec_d = all_out[idx: idx + bsize]
            inp_d = all_in[idx:  idx + bsize]
            idx  += bsize

        # Zero gradient cũ
            self.optimizer.zero_grad()

        # Tính loss mean trên domain
            loss_d = self.criterion(rec_d, inp_d)

        # 2) Lấy gradient trung bình mỗi domain mà không free graph
            grad_list = torch.autograd.grad(
                loss_d,
                list(self.model.parameters()),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,        # cho phép param không tham gia graph
            )

        # 3) Thay None thành zero và flatten
            flat = []
            for g, p in zip(grad_list, self.model.parameters()):
                if g is None:
                    flat.append(torch.zeros_like(p).view(-1))
                else:
                    flat.append(g.contiguous().view(-1))
            grads.append(torch.cat(flat))

    # Cleanup
        self.optimizer.zero_grad()

    # 4) Stack và tính penalty Fishr‐approx
        G      = torch.stack(grads, dim=0)       # [D, P]
        mean_G = G.mean(dim=0, keepdim=True)     # [1, P]
        penalty = ((G - mean_G).pow(2).mean())   # scalar

        return penalty

