import torch
import torch.nn as nn

from phylo_gnn.models.final_layer.abstract_layer import AbstractLayer


class BasicLinearLayer(AbstractLayer):
    def __init__(self, hidden_dim, graph_features, output_dim, **kwargs):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(hidden_dim + graph_features, output_dim)
        )


class FunnelLinearLayer(AbstractLayer):
    def __init__(self, hidden_dim, graph_features, output_dim, depth=3, dropout=0.5, **kwargs):
        assert hidden_dim // (2 ** depth) >= output_dim
        super().__init__()

        layers = [
            nn.Linear(1 * hidden_dim + graph_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
        ]
        current_dim = hidden_dim

        for i in range(depth - 2):
            next_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p=dropout))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.sequential = nn.Sequential(*layers)
