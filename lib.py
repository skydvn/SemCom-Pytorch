'''
Title:    : 
Project   : 
----------------------------------------------------------------------------
Author    : Nguyen Thi Hoai Linh
Email     : linhnth@hn.soc.one
Date      : 2025-07-20 22:23:25
Last Modified : 2025-07-22 19:22:12
Modified By   : Nguyen Thi Hoai Linh
----------------------------------------------------------------------------
Description: 

----------------------------------------------------------------------------
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''



import os
import argparse
from train.train_djsccn import DJSCCNTrainer
from train.train_djsccf import DJSCCFTrainer
from train.train_dgsc import DGSCTrainer
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import time

trainer_map = {
    "djsccf": DJSCCFTrainer,
    "djsccn": DJSCCNTrainer,
    "dgsc": DGSCTrainer,
    }

ratio_list = [1/6]
snr_list = [13]


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_list',nargs='+', default=[],
    help='List of channel domains, e.g. AWGN10 Rayleigh10'
    )
    parser.add_argument('--out', type=str, default='./out',
                         help="Path to save outputs")
    parser.add_argument("--ds", type=str, default='cifar10',
                        help="Dataset")
    parser.add_argument("--base_snr", type=float, default=10,
                        help="SNR during train")
    parser.add_argument('--channel_type', default='AWGN', type=str,
                         help='channel')
    parser.add_argument("--recl", type=str, default='mse',
                        help="Reconstruction Loss")
    parser.add_argument("--clsl", type=str, default='ce',
                        help="Classification Loss")
    parser.add_argument("--disl", type=str, default='kl',
                        help="Invariance and Variance Loss")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Inner learning Rate")

    # Loss Setting
    parser.add_argument("--cls-coeff", type=float, default=0.5,
                        help="Coefficient for Classification Loss")
    parser.add_argument("--rec-coeff", type=float, default=1,
                        help="Coefficient for Reconstruction Loss")
    parser.add_argument("--inv-coeff", type=float, default=0.2,
                        help="Coefficient for Invariant Loss")
    parser.add_argument("--var-coeff", type=float, default=0.2,
                        help="Coefficient for Variant Loss")

    # Model Setting
    parser.add_argument("--inv-cdim", type=int, default=32,
                        help="Channel dimension for invariant features")
    parser.add_argument("--var-cdim", type=int, default=32,
                        help="Channel dimension for variant features")

    # VAE Setting
    parser.add_argument("--vae", action="store_true",
                        help="vae switch")
    parser.add_argument("--kld-coeff", type=float, default=0.00025,
                        help="VAE Weight Coefficient")

    # Meta Setting
    parser.add_argument("--bs", type=int, default=128,
                        help="#batch size")
    parser.add_argument("--wk", type=int, default=os.cpu_count(),
                        help="#number of workers")
    parser.add_argument("--out-e", type=int, default=50,
                        help="#number of epochs")
    parser.add_argument("--dv", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--device", type=bool, default=True,
                        help="Return device or not")
    parser.add_argument("--operator", type=str, default='window',
                        help="Operator for Pycharm")

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
                        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
                        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="SemCom-",
                        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
                        help='toggle to use wandb for online saving')
    parser.add_argument("--verbose", action="store_true",
                        help="printout mode")
    parser.add_argument("--algo", type=str, default="djsccn",
                        help="necst/djsccf mode")
    
    # RUNNING
    parser.add_argument('--train_flag', type=str, default="True",
                        help='Training mode')
    
    parser.add_argument('--num_iter', type=int, default=10,help='Number of iterations for eDJSCC')
    parser.add_argument('--num_channels', type=int, default=16, help='Number of channels')
    parser.add_argument('--num_conv_blocks', type=int, default=2, help='Number of convolutional blocks')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
 

    args = parser.parse_args()
    args.ratio = float(ratio_list[0])

    return args

## Encoder

def encoder_python(image):
    
    args = get_common_args()

    if args.algo not in trainer_map:
        raise ValueError("Invalid trainer")
    
    # Khởi tạo trainer cho bất kỳ args.algo nào
    trainer = DJSCCNTrainer(args)
    model = trainer.model
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ----- đảm bảo image là numpy trước -----
    # Giả sử image là (H, W, C) hoặc (C, H, W)
    if isinstance(image, torch.Tensor):
        input_tensor = image.float()
    else:
        input_tensor = torch.from_numpy(image).float()

    # Chuyển (H, W, C) → (C, H, W) nếu cần
    if input_tensor.ndim == 3 and input_tensor.shape[0] != 3:
        input_tensor = input_tensor.permute(2, 0, 1)

    # Nếu chưa có batch dimension thì thêm
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # [1, 3, 32, 32]

    # Nếu đã có rồi (đôi khi user truyền sẵn [1, 3, 32, 32]) thì giữ nguyên
    elif input_tensor.ndim == 4:
        pass

    else:
        raise ValueError(f"input_tensor có shape bất thường: {input_tensor.shape}")

    input_tensor = input_tensor.to(device)


    # Encode
    with torch.no_grad():
        try:
            encoded_output = model.encoder(input_tensor)
        except Exception as e:
            print("Lỗi khi chạy encoder:", e)
            return None

    # Chuyển về CPU numpy
    encoded_output = encoded_output.detach().cpu().numpy()
    np.savetxt("encoded_output.txt", encoded_output.flatten(), fmt="%.6f")

    return encoded_output


# encode (forward function)
# decode - djsccn.py
def decoder_python(encoded_tensor):
    
    args = get_common_args()

    if args.algo not in trainer_map:
        raise ValueError("Invalid trainer")
    
    TrainerClass = trainer_map[args.algo]
    if args.algo != None :
        args.snr_list = snr_list
        args.ratio = ratio_list
        args.pass_channel = True
        if args.ds == 'cifar10':
            args.image_dims = (3, 32, 32)
            args.downsample = 2
            #args.bs = 128

        # Kích thước latent channels
        args.channel_number = int(args.var_cdim)

        # Unpack spatial dims
        _, H, W = args.image_dims


    # Khởi tạo trainer cho bất kỳ args.algo nào
    trainer = DJSCCNTrainer(args)
    model = trainer.model
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Decode
    if isinstance(encoded_tensor, np.ndarray):
        float_tensor = torch.from_numpy(encoded_tensor.astype(np.float32))
    else:
        float_tensor = encoded_tensor.float()

    # Reshape nếu cần (giả sử encoder output là [1, 32, 8, 8])
    if float_tensor.ndim == 1:
        float_tensor = float_tensor.view(1, 32, 8, 8)
    elif float_tensor.ndim == 2:
        float_tensor = float_tensor.unsqueeze(0)  # [1, C, H, W] 
    elif float_tensor.ndim == 3:
        float_tensor = float_tensor.unsqueeze(0)

    float_tensor = float_tensor.to(device)

    # Decode
    with torch.no_grad():
        decoder_output = model.decoder(float_tensor)


    # Hiển thị ảnh
    output_image = decoder_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_image = np.clip(output_image, 0, 1)

    # Scale ảnh về khoảng [0, 1] dựa trên max hiện tại (không ảnh hưởng đến model)
    output_image_norm = output_image / output_image.max()

    plt.imshow(output_image_norm)
    plt.title("Decoded Image (Scaled)")
    plt.axis("off")
    plt.savefig("decoded_image_scaled.png")

    print("Decoded shape:", output_image.shape)


    # plt.imshow(output_image)
    # plt.title("Decoded Image")
    # plt.axis("off")
    # plt.savefig("decoded_image.png")

    #plt.show()

    return output_image




