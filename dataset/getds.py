from torchvision.datasets import CIFAR10, EMNIST, MNIST, CelebA
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch
import numpy as np
import os


def get_mnist(args):
    path = '/home/duong/data'

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_set = MNIST(root=path, train=True, download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=2)
    test_set = MNIST(root=path, train=False, download=True, transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.tsbs, shuffle=False, num_workers=2)
    valid_set = MNIST(root=path, train=False, download=True, transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.tsbs, shuffle=False, num_workers=2)

    return (train_dl, test_dl, valid_dl), args


def get_emnist(args):
    path = '/home/duong/data'
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    train_set = EMNIST(root=path, split="digits", train=True, download=True, transform=transform)
    test_set = EMNIST(root=path, split="digits", train=False, download=True, transform=transform)
    valid_set = EMNIST(root=path, split="digits", train=False, download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_set, batch_size=args.tsbs, shuffle=False, num_workers=1)
    valid_dl = DataLoader(valid_set, batch_size=args.tsbs, shuffle=False, num_workers=1)

    return (train_dl, test_dl, valid_dl), args


def get_cifar10(args):
    path = '/home/duong/data'
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(148),
            # transforms.Resize(64),
            transforms.ToTensor()
        ]
    )

    train_set = CIFAR10(root=path, train=True, download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=24)
    test_set = CIFAR10(root=path, train=False, download=True, transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.tsbs, shuffle=False, num_workers=24)
    valid_set = CIFAR10(root=path, train=False, download=True, transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.tsbs, shuffle=False, num_workers=24)
    return (train_dl, test_dl, valid_dl), args


def get_cinic10(args):
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=(-1, 1), translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    root_dir = '/home/duong/data/CINIC-10'

    train_set = ImageFolder(root=root_dir + "/train", transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=24)
    valid_set = ImageFolder(root=root_dir + "/valid", transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.tsbs, shuffle=False, num_workers=24)
    test_set = ImageFolder(root=root_dir + "/test", transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.tsbs, shuffle=False, num_workers=24)
    return (train_dl, test_dl, valid_dl), args


def get_dsprites(args):
    class DSprDS(Dataset):
        def __init__(self, split='train', seed=42):
            super().__init__()
            np.random.seed(seed)
            self.root_path = "/".join(
                os.getcwd().split("/")[:-2]) + "/dataset/dsprites/source/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

            self.img_data = np.load(self.root_path, allow_pickle=True, encoding='bytes')['imgs']
            np.random.shuffle(self.img_data)

            self.ori_len = self.img_data.shape[0]

            self.ratio_mapping = {
                "train": (0, int(self.ori_len * 0.95)),
                "valid": (int(self.ori_len * 0.95), int(self.ori_len * 0.975)),
                "test": (int(self.ori_len * 0.975), int(self.ori_len))
            }

            self.split = split
            self.ratio = self.ratio_mapping[split]
            self.data = self.img_data[self.ratio[0]:self.ratio[1]]

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            img = self.data[idx]

            torch_img = torch.from_numpy(img).unsqueeze(0)

            return torch_img.float()

    train_set = DSprDS(split='train')
    train_dl = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=24)
    test_set = DSprDS(split='test')
    test_dl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=24)
    valid_set = DSprDS(split='valid')
    valid_dl = DataLoader(valid_set, batch_size=256, shuffle=False, num_workers=24)
    return (train_dl, test_dl, valid_dl), args


def get_celeb(args):
    path = '/home/duong/data'

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor()
        ]
    )

    train_set = CelebA(root=path, split='train', download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8)
    valid_set = CelebA(root=path, split='valid', download=True, transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.tsbs, shuffle=False, num_workers=8)
    test_set = CelebA(root=path, split='test', download=True, transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.tsbs, shuffle=False, num_workers=8)

    return (train_dl, test_dl, valid_dl), args


def get_ds(args):

    ds_mapping = {
        "mnist": get_mnist,
        "emnist": get_emnist,
        "dsprites": get_dsprites,
        "cifar10": get_cifar10,
        "cinic10": get_cinic10,
        "celeba": get_celeb,
    }

    data, args = ds_mapping[args.ds](args)

    return data, args