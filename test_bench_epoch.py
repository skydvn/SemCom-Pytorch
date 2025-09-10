'''
Title:    : 
Project   : 
----------------------------------------------------------------------------
Author    : Nguyen Thi Hoai Linh
Email     : 
Date      : 2025-09-10 08:28:06
Last Modified : 2025-09-10 08:54:35
Modified By   : Nguyen Thi Hoai Linh
----------------------------------------------------------------------------
Description: 

----------------------------------------------------------------------------
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import torch
import numpy as np
import random
import torch.nn as nn
from models.swinjscc import SWINJSCC
from models.dgsc import DGSC_CIFAR
from dataset.getds import get_cifar10  # Import hàm lấy dataset
from channels.channel_base import Channel  # Import lớp Channel
from utils.data_utils import image_normalization
from utils.metric_utils import get_psnr, view_model_param

from skimage import data
from PIL import Image
import matplotlib.pyplot as plt
from load_lib import * 
import os

class Args:
    base_snr = 20  # Example SNR value
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inv_cdim = 32
    var_cdim = 32
    bs = 1  # Batch size
    ds = "cifar10"  # Dataset name
    snr_list = [10]  # Danh sách SNR
    ratio = 1/6
    #algo = "swinjscc"  # Tên thuật toán
    channel_number = 32
    channel_type = "AWGN"
    image_dims = (3, 32, 32)
    downsample = 2
    encoder_kwargs = dict(
        img_size=(32, 32), patch_size=2, in_chans=3,
        embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=torch.nn.LayerNorm, patch_norm=True
    )
    decoder_kwargs = dict(
        img_size=(32, 32),
        embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=torch.nn.LayerNorm, patch_norm=True
    )
    pass_channel = True 

# Initialize arguments
args = Args()

# Initialize the model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load 1 figure
# -------------------------------

Flag_cifar = 0  # 1 = dùng CIFAR-10, 0 = dùng ảnh ngoài CIFAR
if Flag_cifar == 1:
    # Lấy ảnh từ CIFAR-10 test set
    (train_dl, test_dl, valid_dl), _ = get_cifar10(args)
    data_iter = iter(test_dl)
    images, labels = next(data_iter)

else:
    # Danh sách ảnh RGB trong skimage.data (đều không thuộc CIFAR-10)
    skimage_images = [
        data.chelsea,     # Mèo
        data.astronaut,   # Phi hành gia
        data.coffee,      # Ly cà phê
        data.hubble_deep_field,  # Ảnh thiên văn
    ]
    img_func = random.choice(skimage_images)
    img = img_func()
    img_pil = Image.fromarray(img).resize((32, 32))

    img_tensor = torch.tensor(np.array(img_pil), dtype=torch.float32) / 255.0
    images = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)

    # Label giả cho ảnh ngoài tập (vd: -1), Gán nhãn giả (vì ảnh này không thuộc CIFAR-10)
    labels = torch.tensor([-1])  # -1 = unknown / outside dataset


images, labels = images.to(device), labels.to(device)
# Convert image to dec
print("Images size:", images.size())
print(images.min(), images.max())
dec_tensor = image_to_binary(images, size=(32, 32))
np.savetxt("dec_tensor.txt", dec_tensor, fmt="%d")


# -------------------------------
# chckpoint
# -------------------------------

# Load model from checkpoint
checkpoints = [
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_21h19m33s_on_Sep_09_2025/epoch_4.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_21h09m39s_on_Sep_09_2025/epoch_9.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_21h27m23s_on_Sep_09_2025/epoch_14.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_20h23m39s_on_Sep_09_2025/epoch_19.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_20h41m29s_on_Sep_09_2025/epoch_29.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_19h40m58s_on_Sep_09_2025/epoch_49.pkl",
    "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_15h45m22s_on_Aug_04_2025/epoch_199.pkl",
]
# -------------------------------
# Loop checkpoint 
# -------------------------------
snr_db = 0
for ckpt_path in checkpoints:
    # Initialize the model with only 3 arguments
    model = SWINJSCC(args, 3, 10).to(device)   # 3 channels (RGB), 10 classes (CIFAR-10)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load state_dict into the model
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint, strict=False)  

    for name, param in model.state_dict().items():
        print(f"{name:40} {tuple(param.shape)}")

    model.eval()
    psnr_values = []
    criterion = nn.MSELoss()
    rate = args.channel_number
    model.eval()

    model.change_channel(channel_type=args.channel_type, snr=snr_db)
    feature, mask = model.encode_and_save(dec_tensor, snr_db)
    recon_image = model.channel_and_decode(feature, mask, images, snr_db)
    recon = image_normalization('denormalization')(recon_image)
    gt = image_normalization('denormalization')(images)
    loss = criterion(gt, recon) 
    psnr = get_psnr(image = None, gt = None ,mse = loss)
    
    print(f"SNR: {snr_db} || Test Loss: {loss} || PSNR: {psnr}")
    psnr_values.append(psnr.item())

    # -------------------------------
    # Save figure
    # -------------------------------

    recon_image = recon_image.clamp(0, 1).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    orig_image = images.cpu().squeeze(0).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(orig_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_image)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    # Save 
    ckpt_name = os.path.basename(ckpt_path).replace(".pkl", "")
    save_name = f"reconstruction_{ckpt_name}_snr{snr_db}.png"

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()


