import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 is a classic convolutional neural network architecture that was
    introduced in 1998 by Yann LeCun et al. It was designed for handwritten
    digit recognition and achieved state-of-the-art performance on the MNIST
    dataset at the time.

    Args:
        num_classes (int): The number of classes in the classification task.
            Default is 10, which corresponds to the number of digits in MNIST.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer, which has 6 output
            channels, a kernel size of 5x5, and a padding of 2.
        conv2 (nn.Conv2d): The second convolutional layer, which has 16 output
            channels and a kernel size of 5x5.
        fc1 (nn.Linear): The first fully connected layer, which has 120 output
            features.
        fc2 (nn.Linear): The second fully connected layer, which has 84 output
            features.
        fc3 (nn.Linear): The final fully connected layer, which produces the
            output logits for each class.

    Methods:
        forward(x): Computes the forward pass of the LeNet-5 module on the
            input tensor x. Returns the output logits for each class.

    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
