import torch

def reset_model(model_class, device, **kwargs):
    model = model_class(**kwargs).to(device)
    model.apply(init_weights)  # Apply weight initialization
    return model

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)