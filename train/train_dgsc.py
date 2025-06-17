# from tqdm import tqdm
# import numpy as np
# import os
# import glob

# import torch
# from torch import nn
# from torch.optim import Adam

# from train.train_base import BaseTrainer
# from models.dgsc import *
# from collections import OrderedDict

# from backpack import backpack, extend
# from backpack.extensions import BatchGrad

# def extend_all(module):
#     extend(module)
#     for child in module.children():
#         extend_all(child)
# class DGSCTrainer(BaseTrainer):
#     def __init__(self, args):
#         super().__init__(args)

#         self.model = DGSC_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
#         self.optimizer = Adam(self.model.parameters(), lr=args.lr)
#         self.criterion = nn.MSELoss(reduction='mean')  
#         self.num_domains = 3
#         self.penalty_weight = 0.1      
#         self.ema_decay      = 0.9         
#         self.penalty_anneal_iters  = 5
        
#         self.ema_per_domain = [
#             MovingAverage(ema=self.ema_decay, oneminusema_correction=True)
#             for _ in range(self.num_domains)
#         ]
           
#         self.bce_extended = nn.MSELoss(reduction='none')
#         self.update_count = 0
#         self.domain_list = args.domain_list
#         print(self.domain_list)
#     def parse_domain(self, domain_str):
#         """Extract channel name and SNR from domain string."""
#         channel_name = ''.join([c for c in domain_str if not c.isdigit()])
#         snr = ''.join([c for c in domain_str if c.isdigit()])
#         return channel_name, int(snr)
    
#     def train(self):
#         # for name, m in self.model.encoder.named_modules():
#         #     if isinstance(m, nn.Sequential):
#         #         print(f" Nafd {name}: {m}") 
#         domain_list = self.domain_list 
#         for epoch in range(self.args.out_e):
#             self.model.train()

#             # epoch_loss = 0
#             # epoch_val_loss = 0
#             for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
#                 x, y = x.to(self.device), y.to(self.device)
#                 total_loss = 0
#                 all_in = []
#                 all_out = []
#                 len_minibatches = []
#                 for i, domain_str in enumerate(domain_list):
#                     channel_type, snr = self.parse_domain(domain_str)
#                     out = self.model.channel_perturb(x, channel_type, snr)
#                     # FIXME the tensors should be flattened later
#                     all_in.append(x)  
#                     all_out.append(out)
#                     len_minibatches.append(x.shape[0])

#                 all_in = torch.cat(all_in, dim=0)
#                 all_out = torch.cat(all_out, dim=0)
               
#                 penalty = self.compute_fishr_penalty(all_out, all_in, len_minibatches)
#                 loss = self.criterion(all_out, all_in) # la so thuc nen phai dung mse khong dung cross entropy 
#                 penalty_weight = 0.1 
#                 self.update_count += 1

#                 objective = loss + penalty_weight * penalty
#                 self.optimizer.zero_grad()
#                 objective.backward()
#                 self.optimizer.step()

#                 #return {'loss': objective.item(), 'nll': loss.item(), 'penalty': penalty.item()}
#                 # Backward
#                 total_loss += objective.item()
#             avg_loss = total_loss / len(self.train_dl)
#             self.writer.add_scalar('train/loss', avg_loss, epoch)
#             print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

#             self.model.eval()
#             epoch_val_loss = 0
#             with torch.no_grad():
#                 for test_imgs, test_labels in tqdm(self.test_dl):
#                     test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
#                     self.model.change_channel(channel_type = 'Rayleigh', snr = 13)
#                     test_rec = self.model.forward(test_imgs)
#                     loss = self.criterion(test_imgs, test_rec)
#                     epoch_val_loss += loss.detach().item()
#                 epoch_val_loss /= (len(self.test_dl))
#                 print(f"[Val] Epoch {epoch}: loss = {epoch_val_loss}")
#                 self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)

#             self.save_model(epoch=epoch, model=self.model)

#         self.writer.close()
#         self.save_config()

#     def domain_gen(self,x):
#         domain_list = self.domain_list
#         rec = [[] for _ in domain_list]
#         x.to(self.device)
#         for i, domain_str in enumerate(domain_list):
#             out = self.model.channel_perturb(x, domain_str)
#             rec[i].append(out)
#         print(f"rec: {rec[0].shape}") 
#     def l2_between_dicts(self, dict_1, dict_2):
#         assert len(dict_1) == len(dict_2)
#         dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
#         dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
#         return (
#             torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
#             torch.cat(tuple([t.view(-1) for t in dict_2_values]))
#         ).pow(2).mean()

#     #def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
#     def compute_fishr_penalty(self, all_out, all_in, len_minibatches):
#         grads = []
#         idx = 0
#     # 1) Chia theo từng domain:
#         for i, bsize in enumerate(len_minibatches):
#             rec_d = all_out[idx: idx + bsize]     # slice rec for domain d
#             inp_d = all_in[idx:  idx + bsize]     # slice inp for domain d
#             idx += bsize

#         # 2) Tính gradient trung bình cho domain d
#             self.optimizer.zero_grad()
#             loss_d = self.criterion(rec_d.to(self.device), inp_d.to(self.device))
#             grad_list = torch.autograd.grad(
#             loss_d, self.model.parameters(), retain_graph=True, create_graph=True
#             )
#             flat = torch.cat([g.contiguous().view(-1) for g in grad_list])
#             grads.append(flat)

#         self.optimizer.zero_grad()  # dọn sạch grad sau khi thu hết

#     # 3) Stack và tính penalty
#         G      = torch.stack(grads, dim=0)       # [D, P]
#         mean_G = G.mean(dim=0, keepdim=True)     # [1, P]
#         penalty = ((G - mean_G).pow(2).mean())   # scalar

#         return penalty

#     def _get_grads(self, logits,y):
#         self.optimizer.zero_grad()
#         loss = self.bce_extended(logits, y).sum()
#         # for name, weights in self.model.encoder.named_parameters():
#         #     if not hasattr(weights, 'grad_batch'):
#         #         print(f"[WARN] {name} has not been extended!")

#         with backpack(BatchGrad()):
#             loss.backward(
#                 inputs=list(self.model.decoder.parameters()), retain_graph=True, create_graph=True
#             )

#         dict_grads = OrderedDict()
#         for name, weights in self.model.encoder.named_parameters():
#             if hasattr(weights, "grad_batch"):
#                 dict_grads[name] = weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
#             else:
#                 print(f"[❗WARN] {name} has not been extended! → Có thể gây lỗi `grad_batch`.")    
                
            
        
#         return dict_grads

#     def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
#         # grads var per domain
#         grads_var_per_domain = [{} for _ in range(self.num_domains)]
#         for name, _grads in dict_grads.items():
#             all_idx = 0
#             for domain_id, bsize in enumerate(len_minibatches):
#                 env_grads = _grads[all_idx:all_idx + bsize]
#                 all_idx += bsize
#                 env_mean = env_grads.mean(dim=0, keepdim=True)
#                 env_grads_centered = env_grads - env_mean
#                 grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

#         # moving average
#         for domain_id in range(self.num_domains):
#             grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
#                 grads_var_per_domain[domain_id]
#             )

#         return grads_var_per_domain

#     def _compute_distance_grads_var(self, grads_var_per_domain):

#         # compute gradient variances averaged across domains
#         grads_var = OrderedDict(
#             [
#                 (
#                     name,
#                     torch.stack(
#                         [
#                             grads_var_per_domain[domain_id][name]
#                             for domain_id in range(self.num_domains)
#                         ],
#                         dim=0
#                     ).mean(dim=0)
#                 )
#                 for name in grads_var_per_domain[0].keys()
#             ]
#         )

#         penalty = 0
#         for domain_id in range(self.num_domains):
#             penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
#         return penalty / self.num_domains
    

# class MovingAverage:

#     def __init__(self, ema, oneminusema_correction=True):
#         self.ema = ema
#         self.ema_data = {}
#         self._updates = 0
#         self._oneminusema_correction = oneminusema_correction

#     def update(self, dict_data):
#         ema_dict_data = {}
#         for name, data in dict_data.items():
#             data = data.view(1, -1)
#             if self._updates == 0:
#                 previous_data = torch.zeros_like(data)
#             else:
#                 previous_data = self.ema_data[name]

#             ema_data = self.ema * previous_data + (1 - self.ema) * data
#             if self._oneminusema_correction:
#                 # correction by 1/(1 - self.ema)
#                 # so that the gradients amplitude backpropagated in data is independent of self.ema
#                 ema_dict_data[name] = ema_data / (1 - self.ema)
#             else:
#                 ema_dict_data[name] = ema_data
#             self.ema_data[name] = ema_data.clone().detach()

#         self._updates += 1
#         return ema_dict_data



# class DGSCLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.rec_loss = nn.MSELoss()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

#     def forward(self, args, rec, img):
#         rec_loss = self.rec_loss(rec, img)
#         total_loss = args.rec_coeff * rec_loss

#         return total_loss


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

        self.model = DGSC_CIFAR(self.args, self.in_channel,self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss(reduction= 'mean')
        self.channel_type = args.channel_type 
        self.snr = args.base_snr
        self.model.channel = Channel(channel_type = self.channel_type, snr = self.snr)
    def train(self):
        domain_list = self.domain_list
        rec = [ [] for _ in domain_list ]
        for epoch in range(self.args.out_e):
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            self.model.train()
            for x, y in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                rec = self.model(x)
                loss = self.criterion(x, rec)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.detach().item()
            epoch_train_loss /= (len(self.train_dl))
            print(f"[Train] Epoch: Loss = {epoch_train_loss}")
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)

            # self.model.eval()
            # with torch.no_grad():
            #     #"""In để đảm bảo có channel """
            #     # if self.model.channel is not None:
            #     #     print("Validation... has ")
            #     # else:
            #     #     print("Validation... no channel")
            #     for test_imgs, test_labels in tqdm(self.test_dl):
            #         test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
            #         test_rec,mu, logvar= self.model(test_imgs)
            #         loss = self.model.loss(test_rec, test_imgs, mu,logvar)
            #         epoch_val_loss += loss.detach().item()
            #     epoch_val_loss /= len(self.test_dl)
            #     self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            #     print('Validation Loss:', epoch_val_loss)
            # # Lưu checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()

    def domain_gen(self,x):
        domain_list = self.domain_list
        rec = [[] for _ in domain_list]
        x.to(self.device)
        for i, domain_str in enumerate(domain_list):
            out = self.model.channel_perturb(x, domain_str)
            rec[i].append(out)
        print(f"rec: {rec[0].shape}") 
