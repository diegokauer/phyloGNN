from phylo_gnn.models.modules.default_models import DefaultGAT
from phylo_gnn.models.final_layer import BasicLinearLayer

class PhyloGNN:
    def __init__(
            self,
            output_dim,
            n_classes=8, # OJO
            embedding_dim=64,
            hidden_dim=16,
            use_ensemble_model=True,
            model_checkpoint=None,
            use_early_stopper=True,
            backbone=DefaultGAT,
            final_layer=BasicLinearLayer,
            **model_kwargs
            ):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass