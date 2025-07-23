import torch
import torch.nn as nn
import numpy as np
from models.djsccn import DJSCCN_CIFAR
from dataset.getds import get_cifar10  # Import hàm lấy dataset
from channels.channel_base import Channel  # Import lớp Channel
from utils.data_utils import image_normalization
from utils.metric_utils import get_psnr, view_model_param
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
class Args:
    base_snr = 20  # Example SNR value
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inv_cdim = 32
    var_cdim = 32
    bs = 1  # Batch size
    ds = "cifar10"  # Dataset name
    # snr_list = [10]  # Danh sách SNR
    # ratio = 1/6
    channel_number = 32
    channel_type = "AWGN"
    image_dims = (3, 32, 32)


args = Args()

# Initialize the model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with only 3 arguments
model = DJSCCN_CIFAR(args, 3, 10).to(device)   # 3 channels (RGB), 10 classes (CIFAR-10)

# Load model from checkpoint
checkpoint_path = "C:\SemCom\SemCom_domain_new\SemCom-Pytorch\out\checkpoints\CIFAR10_0.16666666666666666__djsccn_00h19m44s_on_Jul_23_2025\epoch_199.pkl"
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


(train_dl, test_dl, valid_dl), _ = get_cifar10(args)
data_iter = iter(test_dl)
image, label = next(data_iter)


image, label = image.to(device), label.to(device)
input_tensor = image.unsqueeze(0).float()  # shape: [1, 3, 32, 32]

# Hiển thị ảnh
image_np = image[0].permute(1, 2, 0).cpu().numpy()
plt.imshow(image_np)
plt.title(f"Label: {label}")
plt.axis("off")
plt.savefig("input_image.png")



# ENCODER
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
    # Chuyển về CPU numpy
encoded_output = encoded_output.detach().cpu().numpy()
np.savetxt("encoded_output.txt", encoded_output.flatten(), fmt="%.6f")

# In 10 giá trị đầu tiên
encoded_flat = encoded_output.flatten()
print("Encoded min/max:", encoded_output.min(), encoded_output.max())
print("Encoded shape:", encoded_output.shape)


encoded_tensor = encoded_output
#DECODER
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