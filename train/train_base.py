import os
# from tqdm import tqdm
import numpy as np

from dataset import *
from glob import glob
import time
import json
import yaml
import wandb
from tensorboardX import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.metric_utils import get_psnr, view_model_param
from utils.data_utils import image_normalization
from channels.channel_base import Channel
from dataset.getds import get_ds

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.params = vars(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
        self.domain_list = args.domain_list
        self.parallel = False
        if args.train_flag == "True":
            self._setup_dirs()
        self._setup_model()
        self.times = 10
        self.channel_type = args.channel_type

        self.dataset_name = args.ds

        self.batch_size = args.bs
        self.num_workers = args.wk
        
    def _setup_dirs(self):
        out_dir = self.args.out
        domain_str = ''.join(getattr(self, 'domain_list', [])) 
        
        # Tạo tên phaser với định dạng thời gian 17h04m29s_on_Jun_06_2025
        timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')   
        if self.args.algo == 'swinjscc':
            phaser = f"{self.args.ds.upper()}_{self.args.ratio}_{self.args.channel_type}{self.args.base_snr}_{self.args.algo}_{timestamp}"
        else :
            phaser = f"{self.args.ds.upper()}_{self.args.ratio}_{domain_str}_{self.args.algo}_{timestamp}"

        self.root_ckpt_dir = os.path.join(out_dir, 'checkpoints', phaser)
        self.root_log_dir = os.path.join(out_dir, 'logs', phaser)
        # lưu .yaml trực tiếp trong out/configs
        self.root_config_dir = os.path.join(out_dir, 'configs')
        self.writer = SummaryWriter(log_dir=self.root_log_dir)

        # đảm bảo thư mục config tồn tại
        os.makedirs(self.root_config_dir, exist_ok=True)

        # Lọc các đối tượng không tuần tự hóa được
        def filter_non_serializable(obj):
            if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
                return obj
            return str(obj)  

        filtered_params = {k: filter_non_serializable(v) for k, v in self.params.items()}

        for key, value in filtered_params.items():
            print(f"{key}: {value} ({type(value)})")

        try:
            self.writer.add_text('config', json.dumps(filtered_params, indent=4))
        except TypeError as e:
            print(f"Error serializing params: {e}")
            print(f"Filtered params: {filtered_params}")
    
    def _setup_model(self):
        
        (self.train_dl, self.test_dl, self.valid_dl), self.args = get_ds(self.args)

        if self.args.ds == "mnist":
            self.in_channel = 1  
        else:
            self.in_channel = 3
        self.class_num = 10

    def evaluate_epoch(self,snr_chan):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for iter, (images, _) in enumerate(self.test_dl):

                images = images.cuda() if self.parallel and torch.cuda.device_count(
                ) > 1 else images.to(self.device)
                model_out = self.model.forward(images,snr_chan)
                
                # Lấy phần tử đầu tiên từ tuple trả về bởi mô hình
                if isinstance(model_out, tuple):
                    outputs = model_out[0]  # recon_image
                else:
                    outputs = model_out
                if self.args.algo == 'swinjscc' or self.args.algo =='fishr' or self.args.algo  == 'newswin' and self.args.train_flag == 'False':
                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(images)
                    #loss = F.mse_loss(images, outputs) if not self.parallel else F.mse_loss(images, outputs)
                    loss = self.criterion(images, outputs) if not self.parallel else self.criterion(images, outputs)
                if self.args.algo == 'swinjscc' or self.args.algo == 'fishr' or self.args.algo =='newswin':
                    loss = self.criterion(images, outputs) if not self.parallel else self.criterion(
                    images, outputs)
                else:
                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(images)
                    loss = self.criterion( images, outputs) if not self.parallel else self.criterion(self.args,
                    images, outputs)
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)

        return epoch_loss

    def evaluate(self, config_path, output_dir):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            params = config['params']
        channel_type = self.channel_type  
        name = f"{os.path.splitext(os.path.basename(config_path))[0]}_{self.channel_type}"  
        
        eval_dir = os.path.join(output_dir, 'eval', name)  # Evaluation directory includes channel
        writer = SummaryWriter(eval_dir)
        phaser = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join(output_dir, 'checkpoints', phaser)
        pkl_list = glob(os.path.join(checkpoint_dir, '*.pkl'))  # Checkpoint directory remains unchanged

        if not pkl_list:
            print(f"Error: No checkpoint files found in {checkpoint_dir}")
            return

        self.model.load_state_dict(torch.load(pkl_list[-1]))  # Load the latest checkpoint
        print("evaluate")
        
        self.eval_snr(writer)
        writer.close()

    def eval_snr(self, writer):
        snr_list = range(0, 26, 1)
        psnr = 0.0 
        for snr in snr_list:
            psnr = 0.0 
            print(f"model: {self.args.algo} || channel: {self.channel_type} || snr: {snr}")
            self.model.change_channel(self.channel_type, snr)
            test_loss = 0
            for i in range(self.times):
                test_loss += self.evaluate_epoch(snr)     # Check later

            test_loss /= self.times
            psnr = get_psnr(image=None, gt=None, mse=test_loss)
            print(f"Test Loss: {test_loss} || PSNR: {psnr}")
            writer.add_scalar('psnr', psnr, snr)
            if self.args.wandb:
                wandb.log({'psnr': psnr}, step=snr)

    def change_channel(self, snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(self.channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss
    
    def save_config(self):
        print("Saving config of algorithm: " + str(self.args.algo))
        # Lấy phaser từ thư mục checkpoint để đặt tên cho config
        phaser = os.path.basename(self.root_ckpt_dir)
        config_name = f"{phaser}.yaml"
        config_path = os.path.join(self.root_config_dir, config_name)

        # Đảm bảo thư mục config tồn tại
        os.makedirs(self.root_config_dir, exist_ok=True)

        with open(config_path, 'w') as f:
            dict_yaml = {
                'dataset_name': self.args.ds,
                'params': self.params,
                'total_parameters': view_model_param(self.model),
                'domain_list': getattr(self, 'domain_list', [])
            }
            yaml.dump(dict_yaml, f)

    def save_model(self, epoch, model):
        # Tạo tên file checkpoint bao gồm ds, ratio, domain_list, algo, và timestamp
        domain_str = ''.join(getattr(self, 'domain_list', []))
        timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        checkpoint_dir = f"{self.root_ckpt_dir}"
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch}.pkl")
        
        # Xóa các checkpoint cũ hơn 1 epoch
        files = glob(f"{checkpoint_dir}/*.pkl")
        for file in files:
            epoch_nb = int(file.split('_')[-1].split('.')[0])
            if epoch_nb < epoch - 1:
                os.remove(file)