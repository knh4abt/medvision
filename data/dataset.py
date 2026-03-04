import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


class TransformSubset(torch.utils.data.Dataset):
    """Applies a transform to a Subset so train/val can have different augmentations."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    """Get torchvision transforms for train or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders():
    """Create train and validation dataloaders from the dataset directory."""
    full_dataset = datasets.ImageFolder(root=config.DATASET_DIR)

    val_size = int(len(full_dataset) * config.VALID_SPLIT)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_dataset = TransformSubset(train_subset, transform=get_transforms(train=True))
    val_dataset = TransformSubset(val_subset, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
