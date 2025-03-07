import torch
import torch.nn.functional as F
import os
import numpy as np


def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def save_model(model, dir, path):
    os.makedirs(dir, exist_ok=True)
    flag = 1
    while True:
        if os.path.exists(path):
            path = path + '_' + str(flag)
            flag += 1
        else:
            break
    torch.save(model.state_dict(), path)
    print("Model saved in {}".format(path))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_model_param(model):
    total_param = 0

    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    return total_param