import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

import config
from .metrics import get_confusion_matrix


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = get_confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig
