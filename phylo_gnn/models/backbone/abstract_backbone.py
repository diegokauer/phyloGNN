import torch


class AbstractBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
