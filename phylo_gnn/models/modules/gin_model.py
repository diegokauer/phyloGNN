import torch

from phylo_gnn.models.modules.abstract import AbstractModel
from phylo_gnn.models.backbone.gin_backbone import GINRelUBackbone
from phylo_gnn.models.final_layer.linear_layer import BasicLinearLayer


class GINModel(AbstractModel):
    def __init__(self, column_transformer, conv_backbone=GINRelUBackbone, final_layer=BasicLinearLayer, **kwargs):
        super().__init__(column_transformer, **kwargs)
        self.conv_backbone = conv_backbone(**kwargs)
        self.final_layer = final_layer(**kwargs)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self._prepare_features(x)
        x = self.conv_backbone(x, edge_index, batch)
        logit = self.final_layer(x)
        return logit
