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
from phylo_gnn.trainer.train import train_embedding
from phylo_gnn.evaluator.evaluate import evaluate_model
from phylo_gnn.evaluator.plot_results import plot_ROC, plot_PR, plot_loss_curves
from phylo_gnn.models.ensemble_model import MajorityVoteEnsemble

graph_transformers = T.Compose([
    T.ToUndirected(),
    # T.AddSelfLoops(),
    # T.VirtualNode(),
])

cols = [
    "Sex_M",
    "age_group_5 to 8",
    "age_group_Over 8",
    "age_group_Under 5",
    'zscore_label_Normal',
    'zscore_label_Thin',
    "Chao1_16S",
    "Shannon_16S",
    "Simpson_16S",
    "Pielou_16S",
]

data = GraphDataFactory()
train_dataset = data.construct_dataset(data.train, cols, graph_transformer=graph_transformers)
test_dataset = data.construct_dataset(data.test, cols, graph_transformer=graph_transformers)

model_class = DefaultGAT#DefaultPNADefaultGIN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer_class = Adam
criterion = F.binary_cross_entropy_with_logits
metric = roc_auc_score
epochs = 1000
batch_size = 8
n_classes = 8
embedding_dim = 64
hidden_dim = 16
k = 5

column_transformer = {
    "0": AbstractTransformer,
    "1": EmbeddingTransformer,
    "2": OneHotTransformer
}

kwargs = {
    "column_transformer": column_transformer,
    # "graph_transformers": graph_transformers,
    "aggregators": [MeanAggregation(), MaxAggregation(), MinAggregation(), StdAggregation()],
    "deg": PNAConv.get_degree_histogram(DataLoader(train_dataset)),
    "scalers": ["identity", "amplification", "attenuation", "linear"],
    "mode": ["attn"],
    "n_classes": n_classes,
    "num_embeddings": data.node_n,
    "embedding_dim": embedding_dim,
    "act": "elu",
    "train_eps": True,
    "final_layer": BasicLinearLayer,
    "depth": 3,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "input_dim": 1 + embedding_dim + n_classes,
    "num_layers": 15,
    "mlp_num_layers": 3,
    "dropout": 0.3,
    "hidden_dim": hidden_dim,
    "graph_features": 10,
    "output_dim": 1,
}

print(data.node_n)

model = model_class(**kwargs)
# model.load_state_dict(torch.load("../../checkpoints/best_model_exp_4_0.9479.pt").state_dict())
# torch.save(model, "../../checkpoints/corrected_best_model_exp_4_0.9479.pt")
#
# pre_trained_embedding = train_embedding(
#     model_class=model_class,
#     device=device,
#     train_dataset=train_dataset,
#     optimizer_class=optimizer_class,
#     criterion=criterion,
#     epochs=200,
#     batch_size=batch_size,
#     **kwargs
# )
#
# kwargs["embedding_state_dict"] = pre_trained_embedding

model, test_metric, train_trajectories, best_models = train_model_with_kfold_cv(
    model_class=model_class,
    device=device,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer_class=optimizer_class,
    criterion=criterion,
    metric=metric,
    epochs=epochs,
    batch_size=batch_size,
    k=k,
    **kwargs
)

test_score, test_y_true, test_y_pred = evaluate_model(
    model=model,
    val_loader=DataLoader(test_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)
train_score, train_y_true, train_y_pred = evaluate_model(
    model=model,
    val_loader=DataLoader(train_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)

# for i, (train_loss, val_loss) in enumerate(train_trajectories):
#     plot_loss_curves(train_loss, val_loss, log_scale=False, title=f"Training and Validation Loss (Fold {i+1})")

if test_metric >= 0.6:
    torch.save(model, f'../../checkpoints/best_model_exp_7_{test_metric:.4f}.pt')

plot_ROC([test_y_true, train_y_true, test_y_true + train_y_true],
         [test_y_pred, train_y_pred, test_y_pred + train_y_pred],
         ["Test Set", "Train Set", "Complete Set"]
         )

plot_PR([test_y_true, train_y_true, test_y_true + train_y_true],
        [test_y_pred, train_y_pred, test_y_pred + train_y_pred],
        ["Test Set", "Train Set", "Complete Set"]
        )

print("Trying Ensemble model:")

ensemble_model = MajorityVoteEnsemble(model_class, best_models, **kwargs)

test_score, test_y_true, test_y_pred = evaluate_model(
    model=ensemble_model,
    val_loader=DataLoader(test_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)
train_score, train_y_true, train_y_pred = evaluate_model(
    model=ensemble_model,
    val_loader=DataLoader(train_dataset, batch_size=batch_size),
    metric=metric,
    device=device
)

print(f"Averaged model train metric: {train_score:.4f}")
print(f"Averaged model test metric: {test_score:.4f}")

# for i, (train_loss, val_loss) in enumerate(train_trajectories):
#     plot_loss_curves(train_loss, val_loss, log_scale=False, title=f"Training and Validation Loss (Fold {i+1})")

if test_score >= 0.6:
    torch.save(ensemble_model, f'../../checkpoints/best_ensemble_model_exp_7_{test_score:.4f}.pt')

plot_ROC([test_y_true, train_y_true, test_y_true + train_y_true],
         [test_y_pred, train_y_pred, test_y_pred + train_y_pred],
         ["Test Set", "Train Set", "Complete Set"]
         )

plot_PR([test_y_true, train_y_true, test_y_true + train_y_true],
        [test_y_pred, train_y_pred, test_y_pred + train_y_pred],
        ["Test Set", "Train Set", "Complete Set"]
        )