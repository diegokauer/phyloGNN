import torch
import torch.nn as nn

class MajorityVoteEnsemble(nn.Module):
    def __init__(self, model_class, state_dicts: list, *model_args, **model_kwargs):
        """
        Parameters:
            model_class: the class of the model (not an instance)
            state_dicts: a list of state_dicts
            model_args, model_kwargs: arguments to initialize the model
        """
        super().__init__()
        self.models = nn.ModuleList()

        for state_dict in state_dicts:
            model = model_class(*model_args, **model_kwargs)
            model.load_state_dict(state_dict)
            model.eval()  # evaluation mode for inference
            self.models.append(model)

    @torch.no_grad()
    def forward(self, x):
        """
        Averages logits from all models before softmax.
        Returns class predictions.
        """
        logits_list = []
        for model in self.models:
            logits, pre_processed_x, orig_x = model(x)  # shape [batch_size, num_classes]
            logits_list.append(logits)

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)  # shape [batch_size, num_classes]
        return avg_logits, pre_processed_x, orig_x
