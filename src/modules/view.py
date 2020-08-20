import torch
import torch.nn as nn


class View(nn.Module):

    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, tensor) -> torch.Tensor:
        return tensor.view(self.size)
