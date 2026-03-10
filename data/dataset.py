import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

import config


class AlbumentationsDataset(torch.utils.data.Dataset):
    """Wrapper to apply albumentations transforms to an ImageFolder dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


def get_transforms(train=True):
    """Get albumentations transforms for train or validation."""
    if train:
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_dataloaders():
    """Create train and validation dataloaders from the dataset directory."""
    # Load with PIL (no transform yet) so albumentations can handle augmentation
    full_dataset = datasets.ImageFolder(root=config.DATASET_DIR, transform=None)

    # Split into train and validation
    val_size = int(len(full_dataset) * config.VALID_SPLIT)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap with albumentations transforms
    train_dataset = AlbumentationsDataset(train_subset, transform=get_transforms(train=True))
    val_dataset = AlbumentationsDataset(val_subset, transform=get_transforms(train=False))

    persistent = config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader
