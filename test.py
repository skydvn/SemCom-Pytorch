import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from utils import get_psnr

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transformsfrom dataset import Vanilla

from tensorboardX import SummaryWriter
import glob


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def test(args=args):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128

        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise Exception('Unknown dataset')

    name = os.path.splitext(os.path.basename(config_path))[0]
    writer = SummaryWriter(os.path.join(output_dir, 'eval', name))
    model = DeepJSCC(c=c)
    model = model.to(params['device'])
    pkl_list = glob.glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
    model.load_state_dict(torch.load(pkl_list[-1]))
    eval_snr(model, test_loader, writer, params, times)
    writer.close()