'''
Title:    : 
Project   : 
----------------------------------------------------------------------------
Author    : Nguyen Thi Hoai Linh
Email     : 
Date      : 2025-08-17 22:40:39
Last Modified : 2025-08-17 22:40:48
Modified By   : Nguyen Thi Hoai Linh
----------------------------------------------------------------------------
Description: 

----------------------------------------------------------------------------
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
# loader.py
import torch
import numpy as np
import random
import torch.nn as nn
from models.swinjscc import SWINJSCC
from dataset.getds import get_cifar10
from skimage import data
from PIL import Image


class Args:
    base_snr = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inv_cdim = 32
    var_cdim = 32
    bs = 1
    ds = "cifar10"
    snr_list = [10]
    ratio = 1/6
    channel_number = 32
    channel_type = "AWGN"
    image_dims = (3, 32, 32)
    downsample = 2
    encoder_kwargs = dict(
        img_size=(32, 32), patch_size=2, in_chans=3,
        embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, norm_layer=torch.nn.LayerNorm, patch_norm=True
    )
    decoder_kwargs = dict(
        img_size=(32, 32),
        embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, norm_layer=torch.nn.LayerNorm, patch_norm=True
    )
    pass_channel = True


def load_model(checkpoint_path: str, args=None):
    if args is None:
        args = Args()

    device = torch.device(args.device)
    model = SWINJSCC(args, 3, 10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, args


def load_data(flag_cifar: int, args=None):
    """Trả về (images, labels) tensor đã đưa lên device"""
    if args is None:
        args = Args()
    device = torch.device(args.device)

    if flag_cifar == 1:
        (train_dl, test_dl, valid_dl), _ = get_cifar10(args)
        images, labels = next(iter(test_dl))
    else:
        skimage_images = [
            data.chelsea,
            data.astronaut,
            data.coffee,
            data.hubble_deep_field,
        ]
        img_func = random.choice(skimage_images)
        img = img_func()
        img_pil = Image.fromarray(img).resize((32, 32))

        img_tensor = torch.tensor(np.array(img_pil), dtype=torch.float32) / 255.0
        images = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)
        labels = torch.tensor([-1])  # ảnh ngoài tập

    return images.to(device), labels.to(device)
