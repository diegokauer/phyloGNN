import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from tqdm import trange
import copy

from phylo_gnn.trainer.train import train_epochs
from phylo_gnn.evaluator.evaluate import evaluate_model


def train_model_with_kfold_cv(model_class, device, train_dataset, test_dataset, optimizer_class, criterion, metric,
                              epochs, k=5, batch_size=8, **kwargs):
    """Trains a model using k-fold cross-validation and evaluates it on a test dataset.

    Args:
        model_class (torch.nn.Module): The PyTorch model class (not an instance).
        device (torch.device): Device to run the model on.
        train_dataset (list): Training dataset.
        test_dataset (list): Test dataset.
        optimizer_class (torch.optim.Optimizer): Optimizer class (not an instance).
        metric (str): Evaluation metric.
        epochs (int): Number of training epochs.
        k (int, optional): Number of folds for cross-validation. Defaults to 5.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
        **kwargs: Additional arguments for the optimizer and training function.

    Returns:
        None
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    best_models = []
    metrics = []
    train_trajectories = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    target = [data.y for data in train_dataset]

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, target)):
        print(f"Fold {fold + 1}/{k}")

        train_subset = [train_dataset[i] for i in train_idx]
        val_subset = [train_dataset[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class(**kwargs).to(device)
        optimizer = optimizer_class(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])

        train_trajectories.append(train_epochs(
            epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device))

        train_metric, _, _ = evaluate_model(model, train_loader, metric, device)
        val_metric, _, _ = evaluate_model(model, val_loader, metric, device)
        test_metric, _, _ = evaluate_model(model, test_loader, metric, device)

        metrics.append(val_metric)
        print(f"Fold {fold + 1} train: {train_metric:.4f}, val: {val_metric:.4f}, test: {test_metric:.4f}")
        best_models.append(copy.deepcopy(model.state_dict()))

    mean_metric, std_metric = np.mean(metrics), np.std(metrics) * 2.2361
    print(f"\nMetric: {mean_metric:.4f} ± {std_metric:.4f}")

    averaged_model_dict = {}
    for key in best_models[0].keys():
        averaged_model_dict[key] = torch.mean(
            torch.stack([weight[key].float() for weight in best_models]), dim=0
        )
    model = model_class(**kwargs).to(device)
    model.load_state_dict(averaged_model_dict)

    test_metric, _, _ = evaluate_model(model, test_loader, metric, device)
    train_metric, _, _ = evaluate_model(model, DataLoader(train_dataset, batch_size), metric, device)

    print(f"Averaged model train metric: {train_metric:.4f}")
    print(f"Averaged model test metric: {test_metric:.4f}")

    return model, test_metric, train_trajectories, best_models
