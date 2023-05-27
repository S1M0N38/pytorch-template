from pathlib import Path

import torchvision
from torch.utils.data import DataLoader

# It's customary to define three Dataloaders:
# - TrainDataLoader: used to train the model
# - ValDataLoader: used to measure the performance during training
# - TestDataLoader: used to measure the performance at the end of the training
#
# I like to implement as different classes so you can define different dataset,
# transformations (e.g. data augmentation for TrainDataLoader), shuffle or not,
# and so on.


class MNISTTrainDataLoader(DataLoader):
    def __init__(self, path: str, **kwargs):
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
            root=str(Path(path).resolve()),
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
    def __init__(self, path: str, **kwargs):
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
            root=str(Path(path).resolve()),
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


# Sometimes you dont have a test dataset, so you can use the validation,
# but be careful tha the performance you measure can be overestimated
# For example if the validation set is used to take some decision about
# the training process (e.g. hyperparms choices, early stopping, ...).
MNISTTestDataLoader = MNISTValDataLoader


# # Example: custom dataloader from a folder of images
# #
# # Ofter the images are in a folder with subfolders for each class.
# # You can use `torchvision.datasets.ImageFolder` to convert the folder
# # to a pytorch dataset.
#
# class ImageFolderDataLoader(DataLoader):
#     def __init__(self, path: str, **kwargs):
#         transform = torchvision.transforms.Compose(
#             [
#                 # Add here any transforms on image
#                 torchvision.transforms.ToTensor(),
#                 # Add here any transformation on tensor
#             ]
#         )
#         target_transform = torchvision.transforms.Compose(
#             [
#                 # You can also transform labels
#                 ...
#             ]
#         )
#
#         self.dataset = torchvision.datasets.ImageFolder(
#             # Converting to path and then resolve let you use soft link as path str
#             root=str(Path(path).resolve()),
#             transform=transform,
#             target_transform=target_transform,
#         )
#         super().__init__(
#             self.dataset,
#             # Add here argements to DataLoader or add to the toml config file
#             **kwargs,
#         )
