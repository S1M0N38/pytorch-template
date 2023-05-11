import torch
import torchvision


class EfficientNetV2S(torch.nn.Module):
    """
    EfficientNetV2-S is a convolutional neural network architecture that was
    introduced in 2021 by Mingxing Tan et al. in "EfficientNetV2: Smaller Models
    and Faster Training" (https://arxiv.org/abs/2104.00298).

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 1000, which corresponds to the number of classes in
            ImageNet.
        weights (str): Path to the pre-trained weights file. If None, the
            model will be initialized with random weights. Default is None.
        dropout (float): The dropout probability for the fully connected layer
            of the model. Default is 0, which means no dropout will be applied.

    Attributes:
        model (torchvision.models.EfficientNet): The EfficientNetV2-S model from
            the torchvision library, with the final fully connected layer
            replaced by a new one that produces the output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the EfficientNetV2-S module on
            the input tensor x. Returns the output logits for each class.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        weights: str = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_s(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetB3(torch.nn.Module):
    """
    EfficientNetB3 is a convolutional neural network architecture that was
    introduced in 2019 by Mingxing Tan et al. in "EfficientNet: Rethinking Model
    Scaling for Convolutional Neural Networks" (https://arxiv.org/abs/1905.11946)

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 1000, which corresponds to the number of classes in
            ImageNet.
        weights (str): Path to the pre-trained weights file. If None, the
            model will be initialized with random weights. Default is None.
        dropout (float): The dropout probability for the fully connected layer
            of the model. Default is 0, which means no dropout will be applied.

    Attributes:
        model (torchvision.models.EfficientNet): The EfficientNetB3 model from
            the torchvision library, with the final fully connected layer
            replaced by a new one that produces the output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the EfficientNetB3 module on
            the input tensor x. Returns the output logits for each class.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        weights: str = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_b3(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(1536, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetB0(torch.nn.Module):
    """
    EfficientNetB0 is a convolutional neural network architecture that was
    introduced in 2019 by Mingxing Tan et al. in "EfficientNet: Rethinking Model
    Scaling for Convolutional Neural Networks" (https://arxiv.org/abs/1905.11946)

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 1000, which corresponds to the number of classes in
            ImageNet.
        weights (str): Path to the pre-trained weights file. If None, the
            model will be initialized with random weights. Default is None.
        dropout (float): The dropout probability for the fully connected layer
            of the model. Default is 0, which means no dropout will be applied.

    Attributes:
        model (torchvision.models.EfficientNet): The EfficientNetB0 model from
            the torchvision library, with the final fully connected layer
            replaced by a new one that produces the output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the EfficientNetB0 module on
            the input tensor x. Returns the output logits for each class.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        weights: str = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.model(x)
