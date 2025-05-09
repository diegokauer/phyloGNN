import torch
from torch_geometric.nn.models import GIN, GCN, PNA, MLP, GAT
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from phylo_gnn.models.modules.abstract import AbstractModel


class DefaultModel(AbstractModel):
    def __init__(self, column_transformer, conv_backbone, final_layer, model_kwargs, embedding_state_dict=None, **kwargs):
        super().__init__(column_transformer, **kwargs)
        self.conv_backbone = conv_backbone(
            in_channels=kwargs["input_dim"],
            num_layers=kwargs["num_layers"],
            hidden_channels=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            act=kwargs["act"],
            **model_kwargs
        )
        self.final_layer = final_layer(**kwargs)

        if embedding_state_dict:
            print("Embedding pre-loaded...")
            self.column_transformer["1"].transformer.load_state_dict(embedding_state_dict)
            for param in self.column_transformer["1"].parameters():
                param.requires_grad = False


    def forward(self, data):
        x, edge_index, batch, graph_attr = data.x, data.edge_index, data.batch, data.graph_attr
        pre_processed_x = self._prepare_features(x)
        if pre_processed_x.requires_grad:
            pre_processed_x.retain_grad()
        x = self.conv_backbone(x=pre_processed_x, edge_index=edge_index, batch=batch)
        # x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        # x_add = global_add_pool(x, batch)
        # x = torch.cat([x_mean, x_max, x_add, graph_attr], dim=1)
        x = torch.cat([x_mean, graph_attr], dim=1)
        logit = self.final_layer(x)
        return logit, pre_processed_x, x


class DefaultGIN(DefaultModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        super().__init__(conv_backbone=GIN, model_kwargs=model_kwargs, **kwargs)


class DefaultGCN(DefaultModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        super().__init__(conv_backbone=GCN, model_kwargs=model_kwargs, **kwargs)


class DefaultPNA(DefaultModel):
    def __init__(self, **kwargs):
        model_kwargs = {
            "aggregators": kwargs["aggregators"],
            "scalers": kwargs["scalers"],
            "deg": kwargs["deg"]
        }
        super().__init__(conv_backbone=PNA, model_kwargs=model_kwargs, **kwargs)

    def name(self):
        return "DefaultPNA"


class DefaultMLP(DefaultModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        super().__init__(conv_backbone=MLP, model_kwargs=model_kwargs, **kwargs)


class DefaultGAT(DefaultModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        super().__init__(conv_backbone=GAT, model_kwargs=model_kwargs,  **kwargs)