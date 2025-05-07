import torch
from tqdm import trange
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from phylo_gnn.evaluator.evaluate import eval_model_loss, evaluate_model
from phylo_gnn.models.early_stopper import EarlyStopperInMemory


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _, _ = model(batch)
        out = out.squeeze()
        # print(batch.x.shape, model(batch).shape)
        loss = criterion(out, batch.y.squeeze(), reduction='sum')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        samples += len(out) if out.ndim > 0 else 1

    avg_loss_train_loss = total_loss / samples
    return avg_loss_train_loss

def train_epochs(epochs, model, train_loader, val_loader, optimizer, criterion, device, metric=roc_auc_score):
    val_loss_list, train_loss_list = [], []
    early_stopper = EarlyStopperInMemory(patience=200, mode=("max", "min"))
    # print(model.column_transformer["1"].transformer.weight)
    cum_loss = 0
    epoch_progress = trange(epochs, desc="Epochs", leave=True)
    for epoch in epoch_progress:

        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_model_loss(model, val_loader, criterion, device)
        train_loss_list.append(avg_loss)
        val_loss_list.append(val_loss)
        cum_loss += avg_loss
        val_score, _, _ = evaluate_model(
            model=model,
            val_loader=val_loader,
            metric=metric,
            device=device
        )
        epoch_progress.set_postfix(
            epoch_avg_loss=f"{avg_loss:.4f}",
            cum_avg_loss=f"{(cum_loss / (epoch + 1)):.4f}",
            val_loss=f"{val_loss:.4f}",
            val_auc_roc=f"{val_score:.4f}"
        )
        if early_stopper((val_score, val_loss), model, epoch):
            print(f"Early stopping triggered! Epoch: {early_stopper.best_epoch}\nBest Val ROC-AUC: {early_stopper.best_score[0]}\nBest Val Loss {early_stopper.best_score[1]}")
            break
    early_stopper.restore_best_model(model)
    # print(model.column_transformer["1"].transformer.weight)
    return train_loss_list, val_loss_list


def train_embedding(model_class, optimizer_class, train_dataset, criterion, device, batch_size, epochs, **kwargs):
    model = model_class(**kwargs).to(device)
    optimizer = optimizer_class(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    cum_loss = 0
    epoch_progress = trange(epochs, desc="Epochs", leave=True)
    for epoch in epoch_progress:
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        cum_loss += avg_loss
        epoch_progress.set_postfix(
            epoch_avg_loss=f"{avg_loss:.4f}",
            cum_avg_loss=f"{(cum_loss / (epoch + 1)):.4f}"
        )
    embedding = model.column_transformer["1"].transformer
    assert isinstance(embedding, torch.nn.Embedding)
    return embedding.state_dict()