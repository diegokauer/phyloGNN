from torch.nn.functional import one_hot

from phylo_gnn.models.transformer.abstract_transformer import AbstractTransformer


class OneHotTransformer(AbstractTransformer):
    def __init__(self, n_classes, **kwargs):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x, col_idx):
        x = x[:, col_idx].long()
        x = one_hot(x, self.n_classes)
        return x
