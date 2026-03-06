import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config


def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_classification_report(y_true, y_pred):
    """Get a detailed classification report."""
    return classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0)


def get_confusion_matrix(y_true, y_pred):
    """Get the confusion matrix."""
    return confusion_matrix(y_true, y_pred)
