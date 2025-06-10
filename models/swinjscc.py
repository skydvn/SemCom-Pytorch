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
from collections import OrderedDict

from backpack import backpack, extend
from backpack.extensions import BatchGrad



def extend_all(module):
    extend(module)
    for child in module.children():
        extend_all(child)
class SWINJSCCTrainer(BaseTrainer):
    
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
        extend_all(self.model.decoder)

        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss(reduction='mean')  
        self.num_domains = 3
        self.penalty_weight = 0.1       # ví dụ 0.1
        self.ema_decay      = 0.9           # ví dụ 0.9
        self.penalty_anneal_iters  = 5
        # self.model.decoder = extend(self.model.decoder)
        # self.model.encoder = extend(self.model.encoder)
        #self.model = extend(self.model)
        self.ema_per_domain = [
            MovingAverage(ema=self.ema_decay, oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        for name, m in self.model.decoder.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  {name}: {m}")
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
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
            for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
                x, y = x.to(self.device), y.to(self.device)
                total_loss = 0
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
                # if self.update_count >= self.penalty_anneal_iters:
                #     penalty_weight = self.penalty_weight 
                # if self.update_count == self.penalty_anneal_iters != 0:
                # # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # # gradient magnitudes that happens at this step.
                #     penalty_weight = 0
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
                    test_rec,_,_ = self.model.forward(test_imgs,snr)
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

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits,y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        for name, weights in self.model.decoder.named_parameters():
            if not hasattr(weights, 'grad_batch'):
                print(f"[WARN] {name} has not been extended!")
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.model.decoder.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict()
        for name, weights in self.model.decoder.named_parameters():
                    if hasattr(weights, "grad_batch"):
                        dict_grads[name] = weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
                    else:
                         print(f"[❗WARN] {name} has not been extended! → Có thể gây lỗi `grad_batch`.")    
                
            
        
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains
    

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
