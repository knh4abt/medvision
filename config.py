import os
from pathlib import Path

# Paths
DATASET_DIR = Path(os.environ.get("MEDVISION_DATA_DIR", "data/NCT-CRC-HE-100K"))
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("runs")

# Dataset
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
VALID_SPLIT = 0.2

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Model
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
FREEZE_BACKBONE = False

# Reproducibility
SEED = 42
