import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import MeanAggregation, MaxAggregation, MinAggregation, StdAggregation, PNAConv

from phylo_gnn.data_factory.graph_factory import GraphDataFactory
from phylo_gnn.models.transformer.embedding_transformer import EmbeddingTransformer
from phylo_gnn.models.transformer.abstract_transformer import AbstractTransformer
from phylo_gnn.models.transformer.one_hot_transformer import OneHotTransformer
from phylo_gnn.models.final_layer.linear_layer import FunnelLinearLayer, BasicLinearLayer
from phylo_gnn.models.modules.default_models import DefaultGIN, DefaultGCN, DefaultPNA, DefaultGAT, DefaultMLP
from phylo_gnn.trainer.train_kfold_cv import train_model_with_kfold_cv
from phylo_gnn.trainer.train import train_epochs
from phylo_gnn.evaluator.evaluate import evaluate_model
from phylo_gnn.evaluator.plot_results import plot_ROC, plot_PR, plot_loss_curves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer_class = Adam
criterion = F.binary_cross_entropy_with_logits
metric = roc_auc_score
epochs = 500
batch_size = 8

graph_transformers = T.Compose([
    T.ToUndirected(),
    # T.AddSelfLoops(),
    # T.NormalizeFeatures(["x"]),
    # T.VirtualNode(),
])

data = GraphDataFactory()
train_dataset = data.build_sample_specific_no_asv(data.train, graph_transformer=graph_transformers)
val_dataset = data.build_sample_specific_no_asv(data.val, graph_transformer=graph_transformers)
test_dataset = data.build_sample_specific_no_asv(data.test, graph_transformer=graph_transformers)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

column_transformer = {
    "0": AbstractTransformer,
    "1": EmbeddingTransformer,
    "2": OneHotTransformer
}

kwargs = {
    "column_transformer": column_transformer,
    "graph_transformers": graph_transformers,
    "aggregators": [MeanAggregation(), MaxAggregation(), MinAggregation(), StdAggregation()],
    "deg": PNAConv.get_degree_histogram(DataLoader(train_dataset)),
    "scalers": ["identity", "amplification", "attenuation", "linear"],
    "mode": ["attn"],
    "n_classes": 7,
    "num_embeddings": data.node_n,
    "embedding_dim": 64,
    "act": "elu",
    "final_layer": FunnelLinearLayer,
    "depth": 3,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "input_dim": 1 + 64 + 7,
    "num_layers": 3,
    "mlp_num_layers": 3,
    "dropout": 0.3,
    "hidden_dim": 128,
    "graph_features": 13,
    "output_dim": 1,
}

for t in range(100):
    for model_class in [DefaultPNA]:
        # train_model_with_kfold_cv(
        #     model_class=model_class,
        #     device=device,
        #     train_dataset=train_dataset,
        #     test_dataset=test_dataset,
        #     optimizer_class=optimizer_class,
        #     criterion=criterion,
        #     metric=metric,
        #     epochs=epochs,
        #     **kwargs
        # )

        model = model_class(**kwargs).to(device)

        train_loss, val_loss = train_epochs(
            epochs=epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_class(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"]),
            criterion=criterion,
            device=device
        )

        test_score, test_y_true, test_y_pred = evaluate_model(
            model=model,
            val_loader=test_loader,
            metric=metric,
            device=device
        )
        val_score, val_y_true, val_y_pred = evaluate_model(
            model=model,
            val_loader=val_loader,
            metric=metric,
            device=device
        )
        train_score, train_y_true, train_y_pred = evaluate_model(
            model=model,
            val_loader=train_loader,
            metric=metric,
            device=device
        )

        plot_loss_curves(train_loss, val_loss, log_scale=False, title=f"Training and Validation Loss (t={t})")

        # print(model)

        print(t)
        if test_score > 0.75:
            torch.save(model, f"{model.name()}_{test_score:.4f}_{t}.pt")

        print(f"Train AUC-ROC: {train_score}")
        print(f"Test AUC-ROC: {test_score}")