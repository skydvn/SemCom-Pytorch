'''
Title:    : 
Project   : 
----------------------------------------------------------------------------
Author    : Nguyen Thi Hoai Linh
Email     : 
Date      : 2025-08-17 22:40:39
Last Modified : 2025-08-29 16:06:48
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
import cv2


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
    pass_channel = False #True 


def load_model(checkpoint_path: str, args=None):
    # Initialize arguments
    if args is None:
        args = Args()

    # Initialize the model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with only 3 arguments
    model = SWINJSCC(args, 3, 10).to(device)   # 3 channels (RGB), 10 classes (CIFAR-10)

    # Load model from checkpoint
    # checkpoint_path = "out/checkpoints/CIFAR10_0.16666666666666666_AWGN13_swinjscc_15h45m22s_on_Aug_04_2025/epoch_199.pkl"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Kiểm tra nội dung checkpoint
    #print("Checkpoint keys:", checkpoint.keys())

    # Load state_dict into the model
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint, strict=False)  

    # Set model to evaluation mode
    model.eval()

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
    model.eval()
    model.change_channel(channel_type=args.channel_type, snr=8.8)
    
    return model, images

def image_to_binary(image_input, size=(32, 32)):
    import torch

    if isinstance(image_input, str):  
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot find image: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif isinstance(image_input, torch.Tensor):  
        # Torch tensor
        img = image_input.detach().cpu().numpy()
        if img.ndim == 4:   # batch [B, C, H, W]
            img = img[0]    # take first
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

    elif isinstance(image_input, np.ndarray):
        img = image_input
        if img.ndim == 4:   # batch [B, C, H, W]
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

    else:
        raise TypeError("image_input must be a file path, NumPy array, or Torch Tensor.")

    # Ensure uint8 image
    if img.dtype != np.uint8:
        img = (255 * img).clip(0, 255).astype(np.uint8)

    # Resize (OpenCV expects (width, height))
    if img.shape[:2] != (size[1], size[0]):
        img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)

    # Flatten into CHW
    vector = np.transpose(img, (2, 0, 1)).flatten()
    return vector.astype(np.uint8)


def binary_to_image(vector, size=(32, 32)):
    w, h = size
    expected_size = 3 * h * w
    if vector.size != expected_size:
        raise ValueError(f"Expected vector of length {expected_size}, got {vector.size}")
    img = vector.reshape((3, h, w)).transpose(1, 2, 0)
    return img.astype(np.uint8)


# def image_to_binary(image_input, size=(32, 32)):
#     # Nếu input là torch.Tensor
#     if isinstance(image_input, torch.Tensor):
#         # Nếu là tensor 4D (batch, C, H, W) -> lấy ảnh đầu tiên
#         if image_input.dim() == 4:
#             image_input = image_input[0]
#         # Nếu tensor ở dạng (C, H, W), chuyển thành (H, W, C)
#         if image_input.dim() == 3:
#             image_input = image_input.permute(1, 2, 0).cpu().numpy()
#         else:
#             raise TypeError("Unsupported torch.Tensor shape")

#     # Nếu input là PIL.Image
#     if isinstance(image_input, Image.Image):
#         image_input = np.array(image_input)

#     # Nếu input là numpy
#     if isinstance(image_input, np.ndarray):
#         image_input = cv2.resize(image_input, size)
#         # chuyển ảnh sang binary (0 hoặc 1)
#         gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
#         _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
#         return torch.tensor(binary, dtype=torch.float32)

#     raise TypeError("Input must be a NumPy array, PIL.Image, or torch.Tensor")


# def binary_to_image(binary_tensor):
#     if not isinstance(binary_tensor, torch.Tensor):
#         raise TypeError("Input must be a torch.Tensor")

#     # convert sang numpy
#     binary_numpy = binary_tensor.detach().cpu().numpy()

#     # nếu dữ liệu là 0/1 → scale về [0,255]
#     img = (binary_numpy * 255).astype(np.uint8)

#     # nếu ảnh chỉ có 1 kênh → expand thành RGB
#     if img.ndim == 2:
#         img = np.stack([img] * 3, axis=-1)

#     return img


# def tensor_to_numpy_img(tensor):
#     import numpy as np
#     import torch

#     # nếu là torch.Tensor -> đưa về numpy
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.detach().cpu()

#         # Nếu có batch (N, C, H, W) -> lấy ảnh đầu tiên
#         if tensor.dim() == 4:
#             tensor = tensor[0]

#         # (C, H, W) -> (H, W, C)
#         if tensor.dim() == 3:
#             tensor = tensor.permute(1, 2, 0)

#         tensor = tensor.numpy()

#     elif isinstance(tensor, np.ndarray):
#         # Nếu có batch (N, H, W, C) -> lấy ảnh đầu tiên
#         if tensor.ndim == 4:
#             tensor = tensor[0]

#     else:
#         raise TypeError("Input must be torch.Tensor or numpy.ndarray")

#     # Chuẩn hóa về 0-255, uint8
#     tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)

#     return tensor
