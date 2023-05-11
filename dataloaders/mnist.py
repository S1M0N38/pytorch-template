"""
This is a Python class for loading the MNIST dataset for training a machine
learning model. MNIST is a dataset of handwritten digits commonly used for
image classification tasks. This class inherits from the PyTorch `DataLoader`
class and uses the `torchvision` package to apply data augmentation and
normalization to the images. The MNIST dataset is downloaded if it does not
already exist in the specified path.
"""

from pathlib import Path

import torchvision
from torch.utils.data import DataLoader


class MNISTTrainDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            ]
        )
        self.dataset = torchvision.datasets.MNIST(
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


class MNISTValDataLoader(DataLoader):
    def __init__(self, path: Path, **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            ]
        )
        self.dataset = torchvision.datasets.MNIST(
            str(path),
            train=True,
            transform=transforms,
            download=True,
        )

        super().__init__(
            self.dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )


MNISTTestDataLoader = MNISTValDataLoader
