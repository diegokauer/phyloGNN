import torch
import torch.nn.functional as F

def evaluate_model(model, val_loader, metric, device):
    model.eval()
    model.to(device)
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out, _, _ = model(batch)  # Forward pass
            out = out.squeeze()
            pred = F.sigmoid(out).cpu()  # Store predictions (move to CPU)
            predictions += pred.flatten().tolist()
            true_labels += batch.y.cpu().flatten().tolist()  # Store true labels
    return metric(true_labels, predictions), true_labels, predictions


def eval_model_loss(model, val_loader, criterion, device):
    total_loss = 0
    samples = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out, _, _ = model(batch) # Forward pass
            out = out.squeeze()
            samples += len(out) if out.ndim > 0 else 1
            loss = criterion(out, batch.y.squeeze(), reduction='sum', pos_weight=torch.tensor(0.1))
            total_loss += loss.sum().item()
    return total_loss / samples
