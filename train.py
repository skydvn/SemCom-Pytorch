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

    for epoch in range(args.out_e):
        tr_total_loss = 0
        temp_loss = 0
        psnr_train = 0
        model.train()
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

        # Logging can averaging
        log_interface.step(epoch=epoch, test_len=len(test_dl))

    """ Evaluate with Semantic Communication """
    model.eval()
    print("GO")
    snr_min = 0
    snr_max = 33
    snr_step = 3
    for snr in range(snr_min, snr_max, snr_step):
        with torch.no_grad():
            psnr_valid = 0
            for valid_img, valid_labels in tqdm(test_dl):
                valid_img, valid_labels = valid_img.to(device), valid_labels.to(device)
                # SNR -> Noise
                noise = torch.max(valid_img) / (10 ** (snr / 10))
                noise = noise.cpu()
                valid_rec = model.get_semcom_recon(valid_img, noise, device)
                test_loss_dict = criterion(args, valid_img, valid_rec)

                psnr_valid += (20 * torch.log10(torch.max(valid_img) / torch.sqrt(test_loss_dict['rec_loss']))).item()

        print(f"SNR: {snr} "
              f"- PSNR_Valid: {psnr_valid / len(test_dl)}")


def train_djsccf(args: argparse):
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
        in_channel = 2
        class_num = 10
    elif args.ds == "cifar10":
        in_channel = 2
        class_num = 10
    else:
        in_channel = 2
        class_num = 10

    model = DJSCCN_CIFAR(args, in_channel, class_num).to(device=device)
    print(model)
    ukie_optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = DJSCCNLoss()

    # Training
    old_loss_value = 1e26
    old_acc_value = 0
    import random
    for epoch in range(args.out_e):
        tr_total_loss = 0
        temp_loss = 0
        psnr_train = 0
        model.train()

        n_var = random.random() * 0.1

        for img, _ in tqdm(train_dl):
            f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                   std=torch.ones(img[:, 0:1, :].size()) * 0).to(device)
            for layer in range(3):
                img_c = img[:, layer:layer+1, :].to(device)

                x = torch.cat((img_c, img_c + f_noise), dim=1)
                rec = model(x)
                ov_loss_dict = criterion(args, x, rec)
                ukie_optimizer.zero_grad()
                ov_loss_dict["total_loss"].backward()
                ukie_optimizer.step()
                f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                       std=torch.ones(img[:, 0:1, :].size()) * n_var).to(device)

                log_interface(key=f"train/loss/total", value=ov_loss_dict["total_loss"].item())
                log_interface(key=f"train/loss/rec", value=ov_loss_dict["rec_loss"].item())
                log_interface(key=f"train/loss/kld", value=ov_loss_dict["kld_loss"].item())
                log_interface(key=f"train/loss/inv", value=ov_loss_dict["inv_loss"].item())
                log_interface(key=f"train/loss/var", value=ov_loss_dict["var_loss"].item())

                psnr_train += ov_loss_dict['psnr_loss'].item()

        model.eval()
        with torch.no_grad():
            correct = 0
            total, rec_val, kld_val, inv_val, var_val, cls_val, psnr_val = 0, 0, 0, 0, 0, 0
            for test_imgs, test_labels in tqdm(test_dl):
                f_noise = torch.normal(mean=torch.zeros(test_imgs[:, 0:1, :].size()),
                                       std=torch.ones(test_imgs[:, 0:1, :].size()) * 0).to(device)
                for layer in range(3):
                    img_c = test_imgs[:, layer:layer+1, :].to(device, non_blocking=True)
                    x = torch.cat((img_c, img_c + f_noise), dim=1)
                    test_rec = model(x)
                    test_loss_dict = criterion(args, x, test_rec)

                    total += test_loss_dict["total_loss"].item()
                    rec_val += test_loss_dict["rec_loss"].item()
                    kld_val += test_loss_dict["kld_loss"].item()
                    inv_val += test_loss_dict["inv_loss"].item()
                    var_val += test_loss_dict["var_loss"].item()
                    cls_val += test_loss_dict["cls_loss"].item()
                    psnr_val += test_loss_dict["psnr_loss"].item()
            total = total / len(test_dl)
            rec_val = rec_val / len(test_dl)
            kld_val = kld_val / len(test_dl)
            inv_val = inv_val / len(test_dl)
            var_val = var_val / len(test_dl)
            cls_val = cls_val / len(test_dl)
            psnr_val = psnr_val / len(test_dl)
        if args.verbose:
            print(f"rec loss: {total}")

        log_interface(key=f"test/loss/total", value=total)
        log_interface(key=f"test/loss/rec", value=rec_val)
        log_interface(key=f"test/loss/kld", value=kld_val)
        log_interface(key=f"test/loss/inv", value=inv_val)
        log_interface(key=f"test/loss/var", value=var_val)
        log_interface(key=f"test/loss/cls", value=cls_val)
        log_interface(key=f"test/loss/psnr", value=psnr_val)

        # Logging can averaging
        log_interface.step(epoch=epoch, test_len=len(test_dl))

    """ Evaluate with Semantic Communication """
    model.eval()
    print("GO")
    snr_min = -33
    snr_max = 33
    snr_step = 3
    for snr in range(snr_min, snr_max, snr_step):
        with torch.no_grad():
            psnr_valid = 0
            for img, _ in tqdm(test_dl):
                f_noise = torch.normal(mean=torch.zeros(img[:, 0:1, :].size()),
                                       std=torch.ones(img[:, 0:1, :].size()) * 0).to(device)
                for layer in range(3):
                    img_c = img[:, layer:layer+1, :].to(device)
                    valid_img = torch.cat((img_c, img_c + f_noise), dim=1)
                    # SNR -> Noise
                    noise = torch.max(valid_img) / (10 ** (snr / 10))
                    noise = noise.cpu()
                    valid_rec = model.get_semcom_recon(valid_img, noise, device)
                    test_loss_dict = criterion(args, valid_img, valid_rec)

                psnr_valid += (20 * torch.log10(torch.max(valid_img) / torch.sqrt(test_loss_dict['rec_loss']))).item()

        print(f"SNR: {snr} "
              f"- PSNR_Valid: {psnr_valid / len(test_dl)}")


def train_necst(args):
    pass
