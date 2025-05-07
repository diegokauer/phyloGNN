import torch
import torch.nn as nn

from phylo_gnn.models.final_layer.abstract_layer import AbstractLayer


class TransformerLayer(AbstractLayer):
    def __init__(self, hidden_dim, output_dim, heads=8, dropout=0.5, **kwargs):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer_layer(x)
        x = self.fc(x)
        return x
