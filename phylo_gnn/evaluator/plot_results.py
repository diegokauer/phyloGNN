import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns


plt.rcParams["font.family"] = "Verdana"


def plot_ROC(true_labels, predictions, names=[]):
    fig, ax = plt.subplots()
    for i, (y_true, y_pred, name) in enumerate(zip(true_labels, predictions, names)):
        if i < len(true_labels) - 1:
            RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)
        else:
            RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax, plot_chance_level=True)

    # Aesthetic components
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlim([-0.02, 1.02])

    plt.show()


def plot_PR(true_labels, predictions, names=[]):
    fig, ax = plt.subplots()
    for i, (y_true, y_pred, name) in enumerate(zip(true_labels, predictions, names)):
        if i < len(true_labels) - 1:
            PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)
        else:
            PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax, plot_chance_level=True)

    # Aesthetic components
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlim([-0.02, 1.02])

    plt.show()


def plot_loss_curves(train_losses, val_losses, title="Training & Validation Loss", log_scale=True):
    """
    Plots training and validation loss curves with an optional log scale.

    Parameters:
    - train_losses (list or array): Training loss per epoch.
    - val_losses (list or array): Validation loss per epoch.
    - title (str): Title of the plot.
    - log_scale (bool): If True, sets y-axis to log scale.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss", marker="o", linestyle="-")
    plt.plot(val_losses, label="Validation Loss", marker="o", linestyle="-")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    if log_scale:
        plt.yscale("log")  # Apply log scale to y-axis

    plt.show()
