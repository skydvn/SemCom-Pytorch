import os
import argparse
from tqdm import tqdm
import numpy as np

from dataset import *
from utils.log import Log, Model_Info, interpolate
from utils.logging import Logging
from glob import glob
import shutil
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.metric_utils import get_psnr


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_dirs()
        self._setup_model()
        self.times = 10
        self.channel = args.channel

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            assert dataset_name == config['dataset_name']
            params = config['params']
            c = config['inner_channel']

        if dataset_name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), ])
            self.test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                            download=True, transform=transform)
            self.test_loader = DataLoader(self.test_dataset, shuffle=True,
                                     batch_size=params['batch_size'], num_workers=params['num_workers'])

        elif dataset_name == 'imagenet':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128

            self.test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
            self.test_loader = DataLoader(self.test_dataset, shuffle=True,
                                     batch_size=params['batch_size'], num_workers=params['num_workers'])
        else:
            raise Exception('Unknown dataset')

    def _setup_dirs(self):
        runs_dir = os.getcwd() + "/runs"
        os.makedirs(runs_dir, exist_ok=True)
        
        ds_dir = f"{runs_dir}/{self.args.ds}"
        os.makedirs(ds_dir, exist_ok=True)
        
        bs_dir = f"{ds_dir}/{self.args.bs}_{self.args.out_e}"
        os.makedirs(bs_dir, exist_ok=True)
        
        bs_dir_len = len(next(os.walk(bs_dir))[1])
        if bs_dir_len == 0:
            self.exp_dir = f"{bs_dir}/exp0"
        else:
            old_exp_dir = f"{bs_dir}/exp{bs_dir_len - 1}"
            if os.path.exists(old_exp_dir) and len(glob(f"{old_exp_dir}/*.parquet")) < 2:
                shutil.rmtree(old_exp_dir)
                self.exp_dir = old_exp_dir
            else:
                self.exp_dir = f"{bs_dir}/exp{bs_dir_len}"
        os.mkdir(self.exp_dir)
        
        self.best_model_path = f"{self.exp_dir}/best.pt"
        self.last_model_path = f"{self.exp_dir}/last.pt"
        self.log_train_path = f"{self.exp_dir}/train_log.parquet"
        self.log_test_path = f"{self.exp_dir}/test_log.parquet"tơ
        self.config_path = f"{self.exp_dir}/config.json"
    
    def _setup_model(self):
        
        (self.train_dl, self.test_dl, self.valid_dl), self.args = get_ds(self.args)
        self.log_interface = Logging(self.args)

        if self.args.ds == "mnist":
            self.in_channel = 1  
        else:
            self.in_channel = 3
        self.class_num = 10

    # def evaluate_semantic_communication(self):
    #     self.model.eval()
    #     snr_min, snr_max, snr_step = 0, 33, 3
    #     for snr in range(snr_min, snr_max, snr_step):times
    #         with torch.no_grad():
    #             psnr_valid = 0
    #             for valid_img, _ in tqdm(self.test_dl):
    #                 valid_img = valid_img.to(self.device)
    #                 noise = torch.max(valid_img) / (10 ** (snr / 10))
    #                 noise = noise.cpu()
    #                 valid_rec = self.model.get_semcom_recon(valid_img, noise, self.device)
    #                 loss_dict = self.criterion(self.args, valid_img, valid_rec)
    #
    #                 psnr_valid += (20 * torch.log10(torch.max(valid_img) / torch.sqrt(loss_dict['rec_loss']))).item()
    #
    #         print(f"SNR: {snr} - PSNR_Valid: {psnr_valid / len(self.test_dl)}")

    def evaluate_epoch(self):
        test_loss = 0.1
        return test_loss

    def evaluate(self, config_path, output_dir, dataset_name):
        name = os.path.splitext(os.path.basename(config_path))[0]
        writer = SummaryWriter(os.path.join(output_dir, 'eval', name))
        pkl_list = glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
        self.model.load_state_dict(torch.load(pkl_list[-1]))        # Check later
        self.eval_snr(writer, params)
        writer.close()

    def eval_snr(self, writer, param):
        snr_list = range(0, 26, 1)
        for snr in snr_list:
            self.model.change_channel(self.channel, snr)
            test_loss = 0
            for i in range(self.times):
                test_loss += self.evaluate_epoch(self.model, param, self.test_loader)     # Check later

            test_loss /= self.times
            psnr = get_psnr(image=None, gt=None, mse=test_loss)
            writer.add_scalar('psnr', psnr, snr)