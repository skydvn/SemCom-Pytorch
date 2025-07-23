'''
Title:    : 
Project   : 
----------------------------------------------------------------------------
Author    : Nguyen Thi Hoai Linh
Email     : linhnth@hn.soc.one
Date      : 2025-07-22 16:56:39
Last Modified : 2025-07-22 19:22:38
Modified By   : Nguyen Thi Hoai Linh
----------------------------------------------------------------------------
Description: 

----------------------------------------------------------------------------
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lib import *

# Define a transformation to convert PIL images to PyTorch tensors (normalized to [0, 1])
transform = transforms.ToTensor()

# Download and load the CIFAR-10 training dataset
# The dataset will be stored in the './data' directory
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Retrieve the first image and its corresponding label from the dataset
image, label = train_set[0]  # 'image' is a tensor with shape (3, 32, 32)

# Convert the image tensor to a NumPy array and change the channel order to (H, W, C)
input_tensor = image.unsqueeze(0).float()  # shape: [1, 3, 32, 32]

# Hiển thị ảnh
image_np = image.permute(1, 2, 0).numpy()
plt.imshow(image_np)
plt.title(f"Label: {label}")
plt.axis("off")
plt.savefig("input_image.png")

# Encode
encoded_output = encoder_python(input_tensor)
if encoded_output is not None:
    print("Output shape:", encoded_output.shape)
else:
    print("Encoder trả về None.")

# In 10 giá trị đầu tiên
encoded_flat = encoded_output.flatten()
print("Encoded min/max:", encoded_output.min(), encoded_output.max())
print("Encoded shape:", encoded_output.shape)


# print("10 giá trị đầu tiên của encoded_output:")
# print(encoded_flat[:1000])

# decode
output_image = decoder_python(encoded_output)
print("Decoded output min/max:", output_image.min(), output_image.max())

