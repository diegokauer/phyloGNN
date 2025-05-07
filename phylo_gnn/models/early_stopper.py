import copy


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, mode="min"):
        """
        Args:
            patience (int): How many epochs to wait before stopping after no improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (str): "min" for loss (lower is better), "max" for accuracy (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.mode = mode
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        else:
            improvement = (current_score < self.best_score - self.min_delta) if self.mode == "min" else (current_score > self.best_score + self.min_delta)
            if improvement:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop


class EarlyStopperInMemory:
    def __init__(self, patience=5, mode=("max", "min"), max_values=(1.0, None)):
        """
        Args:
            patience (int): Epochs to wait before stopping if no improvement.
            mode (tuple): "max" or "min" for each metric.
            max_values (tuple): Optional theoretical max for each metric (e.g. 1.0 for accuracy).
        """
        self.patience = patience
        self.mode = mode
        self.max_values = max_values
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def _is_better(self, current, best):
        for i, m in enumerate(self.mode):
            cur = current[i]
            bst = best[i]
            max_val = self.max_values[i]

            # Skip comparison if metric is maxed out
            if max_val is not None and bst == max_val and cur == max_val:
                continue

            if m == "max":
                if cur > bst:
                    return True
                elif cur < bst:
                    return False
            elif m == "min":
                if cur < bst:
                    return True
                elif cur > bst:
                    return False
            else:
                raise ValueError(f"Invalid mode: {m}")
        return False

    def __call__(self, current_scores, model, epoch):
        if self.best_score is None:
            self.best_score = current_scores
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            if self._is_better(current_scores, self.best_score):
                self.best_score = current_scores
                self.best_epoch = epoch
                self.counter = 0
                self.best_model_state = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Best model restored from epoch {self.best_epoch} with score {self.best_score}.")

    def get_best_score(self):
        return self.best_score

    def get_best_epoch(self):
        return self.best_epoch


