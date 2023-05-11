"""
This is a Python module containing three classes for loading the iNaturalist
2019 dataset for training, validation, and testing of a machine learning model.
iNaturalist 2019 is a large-scale image classification dataset with over 1
million images of plants, animals, and fungi. The dataset is divided into
training, validation, and testing sets, and this module provides `DataLoader`
classes for each set. The classes use the `torchvision` package to apply data
augmentation and normalization to the images. The image size is resized to
224x224, and the mean and standard deviation used for normalization are
pre-calculated values specific to the iNaturalist 2019 dataset.
"""

from pathlib import Path

import torchvision
from torch.utils.data import DataLoader


class INaturalistTrainDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.454, 0.474, 0.367),
                    std=(0.237, 0.230, 0.249),
                ),
            ]
        )

        self.dataset = torchvision.datasets.ImageFolder(path, transforms)
        super().__init__(
            self.dataset,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )


class INaturalistValDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.454, 0.474, 0.367),
                    std=(0.237, 0.230, 0.249),
                ),
            ]
        )

        self.dataset = torchvision.datasets.ImageFolder(path, transforms)
        super().__init__(
            self.dataset,
            shuffle=False,
            pin_memory=True,
            **kwargs,
        )


class INaturalistTestDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.454, 0.474, 0.367),
                    std=(0.237, 0.230, 0.249),
                ),
            ]
        )
        self.dataset = torchvision.datasets.ImageFolder(path, transforms)
        super().__init__(
            self.dataset,
            shuffle=False,
            pin_memory=True,
            **kwargs,
        )
