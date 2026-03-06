import torch
import random
import numpy as np

import config
from data import get_dataloaders
from models import MedicalImageClassifier
from training import Trainer


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    print(f"Building model: {config.MODEL_NAME}")
    model = MedicalImageClassifier()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(model, train_loader, val_loader, device)
    best_acc = trainer.fit()
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
