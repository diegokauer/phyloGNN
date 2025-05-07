import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from phylo_gnn.data_factory.graph_factory import GraphDataFactory


def pre_process_data(data):
    data = data.to(device)
    x = model._prepare_features(data.x)
    x = model.conv_backbone(x=x, edge_index=data.edge_index, batch=data.batch)
    x = global_mean_pool(x, data.batch)
    inputs = torch.cat([x, data.graph_attr], dim=1)
    inputs.grad = None
    return inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_transformers = T.Compose([
    T.ToUndirected(),
])

model = torch.load("../../checkpoints/best_model_exp_4_0.9479.pt")

data = GraphDataFactory()
train_dataset = data.construct_dataset(data.train, graph_transformer=graph_transformers)
test_dataset = data.construct_dataset(data.test, graph_transformer=graph_transformers)
id2taxa = data.id2taxa

# Seleccionar un subconjunto de datos de entrenamiento como referencia para SHAP
explainer_sample_size = min(100, len(test_dataset))  # Para mantener la eficiencia
testing_data = DataLoader(test_dataset, batch_size=1, shuffle=True)
test_data = next(iter(testing_data)).to(device)
print(test_data.x)
test_data.x.requires_grad_(True)
# test_data.edge_index.requires_grad_(True)
test_data.graph_attr.requires_grad_(True)
print(test_data)
out, pre_processed_x, _ = model(test_data)
out.sum().backward()
print(test_data.graph_attr.grad.data.mean(dim=0))

type_dict = {
    0: 'Root',
    1: 'Kingdom',
    2: 'Phylum',
    3: 'Class',
    4: 'Order',
    5: 'Family',
    6: 'Genus',
    7: 'Species'
}

df = pd.DataFrame(
    {
        'taxa': [id2taxa[idx.item()] for idx in test_data.x[:, 1].cpu()],
        'type': [type_dict[i.item()] for i in test_data.x[:, 2].cpu()],
        'grad_sum': [i.item() for i in pre_processed_x.grad.data[:, 1:65].sum(dim=1).cpu()],
        'grad_mean': [i.item() for i in pre_processed_x.grad.data[:, 1:65].mean(dim=1).cpu()]
    }
)

df = df[df.type == 'Genus']

df.sort_values(by='grad_mean', ascending=False, inplace=True)

print(df.head())

df.sort_values(by='grad_mean', ascending=True, inplace=True)

print(df.head())


#
# for idx, grad in zip(test_data.x[:, 1], pre_processed_x.grad.data[:, 1:65].sum(dim=1)):
#     name = id2taxa[idx.item()]
#     print(f"{name}: {grad}\n")