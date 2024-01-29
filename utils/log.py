import pandas as pd
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt


class Log:
    def __init__(self) -> None:
        self.train_dict = {
            "epoch": [],
            "task": [],
            "inner_epoch": [],
            "cls_loss": [],
            "rec_loss": [],
            "psnr_loss": [],
            "kld_loss": [],
            "inv_loss": [],
            "var_loss": [],
            "irep_loss": [],
            "total_loss": [],
            "accuracy": []
        }

        self.test_dict = {
            "epoch": [],
            "cls_loss": [],
            "rec_loss": [],
            "psnr_loss": [],
            "kld_loss": [],
            "inv_loss": [],
            "var_loss": [],
            "irep_loss": [],
            "total_loss": [],
            "accuracy": []
        }

        self.epoch_cnt = 0

    def add_sp_dict(self, sp_loss_dict, inner_epoch, task):
        self.train_dict["epoch"].append(self.epoch_cnt)
        self.train_dict["task"].append(task)
        self.train_dict["inner_epoch"].append(inner_epoch)
        for key in sp_loss_dict:
            self.train_dict[key].append(sp_loss_dict[key].cpu().item())

    def add_qr_dict(self, qr_loss_dict, task):
        self.train_dict["epoch"].append(self.epoch_cnt)
        self.train_dict["task"].append(task)
        self.train_dict["inner_epoch"].append(-1)
        for key in qr_loss_dict:
            self.train_dict[key].append(qr_loss_dict[key].cpu().item())

    def add_ts_dict(self, ts_loss_dict):
        self.test_dict["epoch"].append(self.epoch_cnt)
        for key in ts_loss_dict:
            self.test_dict[key].append(ts_loss_dict[key].cpu().item())

    def export_parquet(self, train_path, test_path):
        train_log_df = pd.DataFrame(self.train_dict)
        train_log_df.to_parquet(path=train_path)

        test_log_df = pd.DataFrame(self.test_dict)
        test_log_df.to_parquet(path=test_path)

    def step(self):
        self.epoch_cnt += 1


class Model_Info:
    def __init__(self, args, train_root):
        self.info_header = ['Folder Path',  # Model location
                            'Rec Loss', 'Inv Loss',  # Model results 'KLD Loss',
                            'Var Loss', 'CLS Loss', 'CLS Acc', 'Total Loss',
                            'Dataset', 'K Shot', 'K Query',  # Model init info from here
                            'KLD Coeff', 'Inv Coeff', 'Var Coeff',  # Loss settings
                            'o_epochs', 'i_epochs',  # Epoch settings
                            'o_lr', 'i_lr']  # Learning rate settings

        self.args = args
        self.file_path = self._init_model_info(train_root=train_root)

    def _init_model_info(self, train_root):
        file_path = train_root + '/' + "all_training_info.csv"

        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(self.info_header)
        return file_path

    def add_model_info(self, folder_path, result_dict):
        new_row_data = [
            folder_path,
            result_dict["rec_loss"].item(), result_dict["inv_loss"].item(),  # result_dict["kld_loss"].item(),
            result_dict["var_loss"].item(), result_dict["cls_loss"].item(), result_dict["accuracy"].item(),
            result_dict["total_loss"].item(),
            self.args.ds, self.args.ks, self.args.kq,
            self.args.kld_coeff, self.args.inv_coeff, self.args.var_coeff,
            self.args.epochs, self.args.inner_epochs,
            self.args.out_lr, self.args.in_lr
        ]
        # print(new_row_data)
        with open(self.file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_row_data)


def interpolate(autoencoder, x_1, x_2, n=20, path='./results'):
    z_1 = autoencoder.get_latent(x_1)[0]
    z_2 = autoencoder.get_latent(x_2)[0]
    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n * w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i * w:(i + 1) * w] = x_hat.reshape(28, 28)
    fig, ax = plt.subplots(figsize=(15, 2))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)

