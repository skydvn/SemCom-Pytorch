import torch
import torch.nn.functional as F
import numpy as np


def get_psnr(image, gt, max_val=255, mse=None):
    if mse is None:
        mse = F.mse_loss(image, gt)
    mse = torch.tensor(mse) if not isinstance(mse, torch.Tensor) else mse

    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr

def view_model_param(model):
    total_param = 0

    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    return total_param