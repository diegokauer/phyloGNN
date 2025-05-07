import torch.nn as nn
from torch_geometric.nn import GINConv, LayerNorm, global_add_pool
from torch_geometric.nn import MLP

from phylo_gnn.models.backbone.abstract_backbone import AbstractBackbone


class GINRelUBackbone(AbstractBackbone):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=3, mlp_num_layers=3, dropout=0.5, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

        # First layer: input_dim -> hidden_dim
        mlp = MLP([input_dim] + [hidden_dim] * (mlp_num_layers - 1), dropout=dropout)
        self.convs.append(GINConv(mlp))
        self.bns.append(LayerNorm(hidden_dim))

        # Middle backbone: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            mlp = MLP([hidden_dim] * mlp_num_layers, dropout=dropout)
            self.convs.append(GINConv(mlp))
            self.bns.append(LayerNorm(hidden_dim))

    def forward(self, x, edge_index, batch):

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)  # Apply batch normalization
            x = self.relu(x)
            x = self.dropout(x)  # Apply dropout

        x = global_add_pool(x, batch)  # Sum pooling for graph-level representation
        return x
