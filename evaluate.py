"""Evaluate the trained model and generate results artifacts."""

import torch
import matplotlib
matplotlib.use("Agg")

import config
from data import get_dataloaders
from models import MedicalImageClassifier
from utils.metrics import get_classification_report, compute_metrics
from utils.visualization import plot_confusion_matrix


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = MedicalImageClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1} "
          f"(val_acc={checkpoint['val_accuracy']:.4f})")
    return model


@torch.no_grad()
def run_evaluation(model, dataloader, device):
    """Run inference on the full validation set."""
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (only need validation split)
    print("Loading data...")
    _, val_loader = get_dataloaders()
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Load trained model
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    model = load_model(checkpoint_path, device)

    # Run evaluation
    print("\nRunning evaluation...")
    y_true, y_pred = run_evaluation(model, val_loader, device)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    print(f"\nPer-Class Classification Report:")
    print(get_classification_report(y_true, y_pred))

    # Save confusion matrix
    results_dir = "results"
    cm_path = f"{results_dir}/confusion_matrix.png"
    print(f"Saving confusion matrix to {cm_path}...")
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    print("Done.")


if __name__ == "__main__":
    main()
