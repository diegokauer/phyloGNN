from torch.nn import Embedding

from phylo_gnn.models.transformer.abstract_transformer import AbstractTransformer


class EmbeddingTransformer(AbstractTransformer):
    def __init__(self, num_embeddings, embedding_dim=32, **kwargs):
        super().__init__()
        self.transformer = Embedding(num_embeddings, embedding_dim,)

    def forward(self, x, col_idx):
        node_id = x[:, col_idx].long()
        node_id_emb = self.transformer(node_id)
        return node_id_emb
