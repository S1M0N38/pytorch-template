import torch
import torchvision


class ResNet18(torch.nn.Module):
    """
    ResNet-18 is a popular convolutional neural network architecture that was
    introduced in 2015 by Kaiming He et al. "Deep Residual Learning for Image
    Recognition" (https://arxiv.org/abs/1512.03385).

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 1000, which corresponds to the number of classes in
            ImageNet.
        weights (str): Path to the pre-trained weights file. If None, the
            model will be initialized with random weights. Default is None.
        dropout (float): The dropout probability for the fully connected layer
            of the model. Default is 0, which means no dropout will be applied.

    Attributes:
        model (torchvision.models.ResNet): The ResNet-18 model from the
            torchvision library, with the final fully connected layer replaced
            by a new one that produces the output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the ResNet-18 module on the
            input tensor x. Returns the output logits for each class.

    """

    def __init__(
        self,
        num_classes: int = 1000,
        weights: str = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)
