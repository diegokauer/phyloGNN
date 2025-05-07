import torch
import torch.nn as nn


class AbstractTransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transformer = nn.Identity()

    def forward(self, x, col_idx):
        x = self.transformer(x[:, col_idx]).unsqueeze(1)
        return x
