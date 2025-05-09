import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from phylo_gnn.data_factory.graph_factory import GraphDataFactory
from phylo_gnn.trainer.train import train_epochs
from phylo_gnn.models.modules.default_models import DefaultGAT
from phylo_gnn.models.final_layer import BasicLinearLayer
from phylo_gnn.models.transformer.embedding_transformer import EmbeddingTransformer
from phylo_gnn.models.transformer.abstract_transformer import AbstractTransformer
from phylo_gnn.models.transformer.one_hot_transformer import OneHotTransformer


class PhyloGNN:
    def __init__(
            self,
            output_dim=1,
            n_classes=8,
            embedding_dim=64,
            hidden_dim=16,
            # use_ensemble_model=False,
            model_checkpoint=None,
            backbone=DefaultGAT,
            head=BasicLinearLayer,
            head_depth=3,
            dropout=0.3,
            **model_kwargs
            ):

        self.backbone = backbone

        column_transformer = {
            "0": AbstractTransformer,
            "1": EmbeddingTransformer,
            "2": OneHotTransformer
        }

        self.kwargs = {
            # Pre-processing
            "n_classes": n_classes,
            "column_transformer": column_transformer,

            # GNN backbone shape parameters
            "num_layers": 15,
            "embedding_dim": embedding_dim,
            "input_dim": 1 + embedding_dim + n_classes,
            "hidden_dim": hidden_dim,
            "dropout": dropout,

            # Classification Layer shape parameters
            "final_layer": head,
            "depth": head_depth,
            "output_dim": output_dim,

            # GAT model parameters
            "act": "elu",
            "train_eps": True,
        }
        self.kwargs.update(model_kwargs)

        if model_checkpoint:
            self.model = torch.load(model_checkpoint)
        else:
            self.model = None

    def train(self, data, train_dataset, validation_dataset, epochs=1000, batch_size=4, lr=1e-3, weight_decay=1e-5):
        kwargs = {
            # Pre-processing
            "graph_transformers":  T.Compose([
                T.ToUndirected(),
            ]),

            # Model parameters based on dataset
            "num_embeddings": data.node_n,
            "graph_features": data.n_graph_features,
        }
        self.kwargs.update(kwargs)
        if self.model is None:
            self.model = self.backbone(**self.kwargs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = F.binary_cross_entropy_with_logits
        metric = roc_auc_score
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(device)

        train_epochs(
            epochs=epochs,
            model=self.model,
            train_loader=train_loader,
            val_loader=validation_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            metric=metric,
        )

    def predict(self, dataset):
        predictions = []
        dataloader = DataLoader(dataset)
        self.model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                out, _, _ = self.model(batch)  # Forward pass
                out = out.squeeze()
                pred = F.sigmoid(out).cpu()  # Store predictions (move to CPU)
                predictions += pred.flatten().tolist()

        return predictions


    def score(self, dataset):
        pass


#
# model = PhyloGNN()
# data = GraphDataFactory()
# subset_1 = data.construct_dataset(data.get_ids([1, 2, 3, 4]), cols=["bmi_zscore", 'Shannon_16S'], taxa_dataframes=[data.taxa_data_bacteria])
# subset_2 = data.construct_dataset(data.get_ids([5, 6, 7, 8]), cols=["bmi_zscore", 'Shannon_16S'], taxa_dataframes=[data.taxa_data_bacteria])
# model.train(data, train_dataset=subset_1, validation_dataset=subset_2)
