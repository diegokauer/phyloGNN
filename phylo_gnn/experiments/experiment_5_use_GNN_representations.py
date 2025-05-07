import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from phylo_gnn.data_factory.graph_factory import GraphDataFactory


# def pre_process_data(data):
#     data = data.to(device)
#     x = model._prepare_features(data.x)
#     x = model.conv_backbone(x=x, edge_index=data.edge_index, batch=data.batch)
#     x = global_mean_pool(x, data.batch)
#     inputs = torch.cat([x, data.graph_attr], dim=1)
#     inputs.grad = None
#     return inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_transformers = T.Compose([
    T.ToUndirected(),
])

model = torch.load("../../checkpoints/best_ensemble_model_exp_7_0.6146.pt").to(device)
model.eval()

data = GraphDataFactory()
train_dataset = data.construct_dataset(data.train, graph_transformer=graph_transformers)
test_dataset = data.construct_dataset(data.test, graph_transformer=graph_transformers)
id2taxa = data.id2taxa

# Seleccionar un subconjunto de datos de entrenamiento como referencia para SHAP
explainer_sample_size = min(100, len(test_dataset))  # Para mantener la eficiencia
testing_data = DataLoader(test_dataset, batch_size=explainer_sample_size, shuffle=True)
test_data = next(iter(testing_data)).to(device)

explainer_sample_size = min(100, len(train_dataset))  # Para mantener la eficiencia
training_data = DataLoader(train_dataset, batch_size=explainer_sample_size, shuffle=True)
train_data = next(iter(training_data)).to(device)

with torch.no_grad():
    _, _, test_data_encoded = model(test_data)
    _, _, train_data_encoded = model(train_data)

test_data_encoded = test_data_encoded.cpu().numpy()
test_y = test_data.y.cpu().numpy()

train_data_encoded = train_data_encoded.cpu().numpy()
train_y = train_data.y.cpu().numpy()

clf = LogisticRegression()
clf.fit(train_data_encoded, train_y)

print(roc_auc_score(y_true=test_y, y_score=clf.predict_proba(test_data_encoded)[:, 1]))

clf.fit(train_data.graph_attr.cpu().numpy(), train_y)

print(roc_auc_score(y_true=test_y, y_score=clf.predict_proba(test_data.graph_attr.cpu().numpy())[:, 1]))

clf.fit(train_data_encoded[:, :16], train_y)

print(roc_auc_score(y_true=test_y, y_score=clf.predict_proba(test_data_encoded[:, :16])[:, 1]))





