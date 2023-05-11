"""
This module provides PyTorch DataLoaders for the CIFAR-100 dataset in training
and validation modes. CIFAR-100 is a dataset of 50,000 32x32 color images in
100 classes, with 500 images per class. The `CIFAR100TrainDataLoader` applies
data augmentation techniques such as random horizontal flipping and
normalization to the input images, while the `CIFAR100ValDataLoader` only
applies normalization. Both DataLoaders use the torchvision library to load
the dataset and return batches of images and their corresponding labels.
"""

from torch.utils.data import DataLoader
import torchvision
from pathlib import Path


class CIFAR100TrainDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        self.dataset = torchvision.datasets.CIFAR100(
            str(path),
            train=True,
            transform=transforms,
            download=True,
        )

        super().__init__(
            self.dataset,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )


class CIFAR100ValDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        self.dataset = torchvision.datasets.CIFAR100(
            str(path),
            train=False,
            transform=transforms,
            download=True,
        )

        super().__init__(
            self.dataset,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )


CIFAR100TestDataLoader = CIFAR100ValDataLoader
