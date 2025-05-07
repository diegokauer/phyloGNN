import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from phylo_gnn.data_factory.graph_factory import GraphDataFactory
from phylo_gnn.models.transformer.embedding_transformer import EmbeddingTransformer
from phylo_gnn.models.transformer.abstract_transformer import AbstractTransformer
from phylo_gnn.models.transformer.one_hot_transformer import OneHotTransformer
from phylo_gnn.models.backbone.gin_backbone import GINRelUBackbone
from phylo_gnn.models.final_layer.linear_layer import FunnelLinearLayer, BasicLinearLayer
from phylo_gnn.models.modules.gin_model import GINModel
from phylo_gnn.trainer.train_kfold_cv import train_model_with_kfold_cv
from phylo_gnn.trainer.train import train_epochs
from phylo_gnn.evaluator.evaluate import evaluate_model
from phylo_gnn.evaluator.plot_results import plot_ROC, plot_PR, plot_loss_curves

graph_transformers = T.Compose([
    T.ToUndirected(),
    # T.AddSelfLoops(),
    # T.VirtualNode(),
])

data = GraphDataFactory()
train_dataset = data.build_sample_specific_dataset(data.train, graph_transformer=graph_transformers)
val_dataset = data.build_sample_specific_dataset(data.val, graph_transformer=graph_transformers)
test_dataset = data.build_sample_specific_dataset(data.test, graph_transformer=graph_transformers)

model_class = GINModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer_class = Adam
criterion = F.binary_cross_entropy_with_logits
metric = roc_auc_score
epochs = 5000
batch_size = 4

column_transformer = {
    "0": AbstractTransformer,
    "1": EmbeddingTransformer,
    "2": OneHotTransformer
}

kwargs = {
    "column_transformer": column_transformer,
    "graph_transformers": graph_transformers,
    "n_classes": 8,
    "num_embeddings": data.node_n,
    "embedding_dim": 64,
    "conv_layer": GINRelUBackbone,
    "final_layer": FunnelLinearLayer,
    "depth": 3,
    "lr": 1e-5,
    "weight_decay": 1e-6,
    "input_dim": 1 + 64 + 8,
    "num_layers": 3,
    "mlp_num_layers": 3,
    "dropout": 0.2,
    "hidden_dim": 512,
    "output_dim": 1,
}

#
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

print(f"Model size: {sum(p.numel() for p in model.parameters())}")

train_loss, val_loss = train_epochs(
    epochs=epochs,
    model=model,
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    optimizer=optimizer_class(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"]),
    criterion=criterion,
    device=device
)

plot_loss_curves(train_loss, val_loss, log_scale=False)


# model = torch.load("best_model_exp_1_ROC_0.7045 - best model.pt")

test_score, test_y_true, test_y_pred = evaluate_model(
    model=model,
    val_loader=DataLoader(test_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)
val_score, val_y_true, val_y_pred = evaluate_model(
    model=model,
    val_loader=DataLoader(val_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)
train_score, train_y_true, train_y_pred = evaluate_model(
    model=model,
    val_loader=DataLoader(train_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)

# print(model)

print(f"Train AUC-ROC: {train_score}")
print(f"Test AUC-ROC: {test_score}")
print(val_y_pred)
print(val_y_true)
print(train_y_pred)
print(train_y_true)

if test_score >= 0.6:
    torch.save(model, f"best_model_exp_1_ROC_{test_score:.4f}.pt")

plot_ROC([test_y_true, train_y_true, val_y_true, test_y_true + train_y_true + val_y_true],
         [test_y_pred, train_y_pred, val_y_pred, test_y_pred + train_y_pred + val_y_pred],
         ["Test Set", "Train Set", "Val Set", "Complete Set"]
         )

plot_PR([test_y_true, train_y_true, val_y_true, test_y_true + train_y_true + val_y_true],
        [test_y_pred, train_y_pred, val_y_pred, test_y_pred + train_y_pred + val_y_pred],
        ["Test Set", "Train Set", "Val Set", "Complete Set"]
        )