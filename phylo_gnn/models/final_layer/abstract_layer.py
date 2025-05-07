import torch


class AbstractLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential()

    def forward(self, x):
        return self.sequential(x)
