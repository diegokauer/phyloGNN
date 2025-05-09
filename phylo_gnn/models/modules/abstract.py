import torch


class AbstractModel(torch.nn.Module):
    def __init__(self, column_transformer, **kwargs):
        super().__init__()
        self.column_transformer = torch.nn.ModuleDict({
            k: column_transformer[k](**kwargs) for k in column_transformer
        })

    def _prepare_features(self, x):
        transformed_features = []
        columns = x.shape[1]

        for col_idx in range(columns):
            transformer = self.column_transformer[str(col_idx)]
            transformed_features.append(transformer(x, col_idx))

        x = torch.cat(transformed_features, dim=1)
        return x

    def forward(self, x):
        pass
