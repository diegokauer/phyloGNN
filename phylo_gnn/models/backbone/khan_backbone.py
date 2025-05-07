import torch.nn as nn
from torch_geometric.nn import GCNConv

from phylo_gnn.models.backbone.abstract_backbone import AbstractBackbone


class KhanBackbone(AbstractBackbone):
    def __init__(self, input_dim=1, output_dim=1, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.elu = nn.ELU
        self.convs = nn.ModuleList([
            GCNConv(input_dim, 64),
            GCNConv(64, 64),
            GCNConv(64, output_dim)
        ])

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)

        return x
