import os
import argparse
from tqdm import tqdm
import numpy as np

from dataset import *
from utils.log import Log, Model_Info, interpolate
from utils.logging import Logging
from datetime import datetime
from glob import glob
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import EMNIST
from torchvision import transforms
from models import *


def train_djsccn(args: argparse):
    # Folder Setup
    runs_dir = os.getcwd() + "/runs"
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    ds_dir = runs_dir + f"/{args.ds}"
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    bs_dir = ds_dir + f"/{args.bs}_{args.out_e}"
    if not os.path.exists(bs_dir):
        os.mkdir(bs_dir)
    bs_dir_len = len(next(os.walk(bs_dir))[1])
    if bs_dir_len == 0:
        exp_dir = bs_dir + "/exp0"
    else:
        old_exp_dir = bs_dir + f"/exp{bs_dir_len - 1}"
        if os.path.exists(old_exp_dir) and len(glob(old_exp_dir + "/*.parquet")) < 2:
            shutil.rmtree(path=old_exp_dir)
            exp_dir = old_exp_dir
        else:
            exp_dir = bs_dir + f"/exp{bs_dir_len}"
    os.mkdir(exp_dir)

    # Path settings
    best_model_path = exp_dir + f"/best.pt"
    last_model_path = exp_dir + f"/last.pt"
    log_train_path = exp_dir + "/train_log.parquet"
    log_test_path = exp_dir + "/test_log.parquet"
    config_path = exp_dir + "/config.json"

    coeff_dict = {
        "epoch": 1,
        "cls_loss": args.cls_coeff,
        "rec_loss": args.rec_coeff,
        "psnr_loss": 1,
        "kld_loss": args.kld_coeff,
        "inv_loss": args.inv_coeff,
        "var_loss": args.var_coeff,
        "irep_loss": 1,
        "total_loss": 1,
        "accuracy": 1
    }

    # Setup
    log_interface = Logging(args)

    (train_dl, test_dl, valid_dl), args = get_ds(args)

    print(len(train_dl), len(test_dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    if args.ds == "emnist":
        in_channel = 1
        class_num = 10
    elif args.ds == "cifar10":
        in_channel = 3
        class_num = 10
    else:
        in_channel = 3
        class_num = 10

    model = DJSCCN_CIFAR(args, in_channel, class_num).to(device=device)
    print(model)
    ukie_optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = DJSCCNLoss()

    # Training
    old_loss_value = 1e26
    old_acc_value = 0
    tr_total_loss = 0
    temp_loss = 0
    psnr_train = 0
    model.train()
    for _ in range(args.out_e):
        for x, y in tqdm(train_dl):
            x, y = x.to(device), y.to(device)
            rec = model(x)

            ov_loss_dict = criterion(args, x, rec)

            ukie_optimizer.zero_grad()
            ov_loss_dict["total_loss"].backward()
            ukie_optimizer.step()

            log_interface(key=f"train/loss/total", value=ov_loss_dict["total_loss"].item())
            log_interface(key=f"train/loss/rec", value=ov_loss_dict["rec_loss"].item())
            log_interface(key=f"train/loss/kld", value=ov_loss_dict["kld_loss"].item())
            log_interface(key=f"train/loss/inv", value=ov_loss_dict["inv_loss"].item())
            log_interface(key=f"train/loss/var", value=ov_loss_dict["var_loss"].item())

            psnr_train += ov_loss_dict['psnr_loss'].item()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for test_imgs, test_labels in tqdm(test_dl):
            test_imgs = test_imgs.to(device, non_blocking=True)
            test_labels = test_labels.to(device, non_blocking=True)
            test_rec = model(test_imgs)
            test_loss_dict = criterion(args, test_imgs, test_rec)

            total += test_labels.size(0)

    if args.verbose:
        print(" - ".join([f"{key}: {round(test_loss_dict[key].item(), 5)}" for key in test_loss_dict]))
        print(" - ".join(
            [f"{key}: {round(test_loss_dict[key].item() * coeff_dict[key], 5)}" for key in test_loss_dict]))

    log_interface(key=f"test/loss/total", value=test_loss_dict["total_loss"].item())
    log_interface(key=f"test/loss/rec", value=test_loss_dict["rec_loss"].item())
    log_interface(key=f"test/loss/kld", value=test_loss_dict["kld_loss"].item())
    log_interface(key=f"test/loss/inv", value=test_loss_dict["inv_loss"].item())
    log_interface(key=f"test/loss/var", value=test_loss_dict["var_loss"].item())
    log_interface(key=f"test/loss/cls", value=test_loss_dict["cls_loss"].item())


def train_djsccf(args):
    pass


def train_necst(args):
    pass
