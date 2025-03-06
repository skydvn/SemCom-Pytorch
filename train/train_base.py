import os
import argparse
from tqdm import tqdm
import numpy as np

from dataset import *
from utils.log import Log, Model_Info, interpolate
from utils.logging import Logging
from glob import glob
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_dirs()
        self._setup_model()

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
        self.log_test_path = f"{self.exp_dir}/test_log.parquet"
        self.config_path = f"{self.exp_dir}/config.json"
    
    def _setup_model(self):
        
        (self.train_dl, self.test_dl, self.valid_dl), self.args = get_ds(self.args)
        self.log_interface = Logging(self.args)

        if self.args.ds == "emnist":
            self.in_channel = 1  
        else:
            self.in_channel = 3
        self.class_num = 10
        
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)