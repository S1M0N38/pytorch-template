import torch

# Deine here losses that you can use config file
# Loss must be implemented as a torch.nn.Module and not as a function

# Use can use losses from torch.nn, for example:
CrossEntropyLoss = torch.nn.CrossEntropyLoss

# # You can also define your own losses, for example:
# class CosineDistanceLoss(torch.nn.Module):
#     """
#     Computes the cosine distance loss between two input tensors.
#
#     Cosine distance measures the dissimilarity between two vectors by computing
#     1 minus the cosine similarity between them. The cosine similarity is a value
#     between -1 and 1, where higher values indicate greater similarity.
#
#     Args:
#         None
#
#     Returns:
#         torch.Tensor: The computed cosine distance loss between the input tensors.
#
#     Shape:
#         - x (torch.Tensor): Input tensor with shape (batch_size, features).
#         - y (torch.Tensor): Input tensor with shape (batch_size, features).
#         - Output (torch.Tensor): Computed cosine distance loss with shape (1,).
#
#     Examples:
#         >>> criterion = CosineDistanceLoss()
#         >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#         >>> y = torch.tensor([[7, 8, 9], [10, 11, 12]])
#         >>> loss = criterion(x, y)
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the cosine distance loss.
#
#         Args:
#             x (torch.Tensor): Input tensor with shape (batch_size, features).
#             y (torch.Tensor): Input tensor with shape (batch_size, features).
#
#         Returns:
#             torch.Tensor: The computed cosine distance loss between the input tensors.
#         """
#         return 1 - torch.nn.functional.cosine_similarity(x, y).mean()
