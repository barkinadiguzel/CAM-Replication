import torch
import torch.nn as nn

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean(dim=(2,3))
