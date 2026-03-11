import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

import config
from utils.metrics import compute_metrics


class Trainer:
    """Handles the training and evaluation loop with mixed precision support."""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = device.type == "cuda"

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        self.best_val_acc = 0.0
        self.patience_counter = 0

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)

        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/train", metrics["accuracy"], epoch)

        return epoch_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]"):
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)

        self.writer.add_scalar("Loss/val", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/val", metrics["accuracy"], epoch)

        return epoch_loss, metrics

    def fit(self):
        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{config.NUM_EPOCHS} — "
                f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )

            # Checkpoint best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_accuracy": self.best_val_acc,
                    },
                    config.CHECKPOINT_DIR / "best_model.pth",
                )
                print(f"  Saved best model (val_acc={self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.writer.close()
        return self.best_val_acc
