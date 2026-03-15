# MedVision

Histology tissue classifier for colorectal cancer. Uses EfficientNet-B0 (pretrained on ImageNet) to classify
H&E-stained tissue patches into 9 classes.

Built with PyTorch and timm.

## Dataset

[NCT-CRC-HE-100K](https://zenodo.org/record/1214456) — 100k histological images (224x224) of human colorectal
cancer tissue, 9 tissue classes:

- **ADI** — Adipose
- **BACK** — Background
- **DEB** — Debris
- **LYM** — Lymphocytes
- **MUC** — Mucus
- **MUS** — Smooth muscle
- **NORM** — Normal colon mucosa
- **STR** — Cancer-associated stroma
- **TUM** — Colorectal adenocarcinoma epithelium

## Results

Trained on 80/20 split, early stopped at epoch 15/30.

**99.6% validation accuracy**

| Class | F1 Score |
|-------|----------|
| ADI   | 1.00     |
| BACK  | 1.00     |
| DEB   | 1.00     |
| LYM   | 1.00     |
| MUC   | 0.99     |
| MUS   | 1.00     |
| NORM  | 1.00     |
| STR   | 0.99     |
| TUM   | 0.99     |

![Confusion Matrix](results/confusion_matrix.png)

## Setup

```bash
pip install -r requirements.txt
```

Set dataset path:

```bash
export MEDVISION_DATA_DIR="/path/to/NCT-CRC-HE-100K"
```

The directory should have the class subfolders directly inside it (`ADI/`, `BACK/`, ..., `TUM/`).

## Usage

Train:

```bash
python main.py
```

Evaluate (prints classification report, saves confusion matrix):

```bash
python evaluate.py
```

TensorBoard:

```bash
tensorboard --logdir runs
```

All hyperparameters are in `config.py`.

## Architecture

- EfficientNet-B0 backbone via [timm](https://github.com/huggingface/pytorch-image-models), pretrained on ImageNet
- Dropout (0.3) + linear classification head
- AdamW with cosine annealing LR schedule
- Mixed precision training (fp16 on CUDA)
- Data augmentation with albumentations (flips, rotations, color jitter, blur)
- Early stopping with patience of 5 epochs

## Project structure

```
medvision/
├── config.py              # hyperparameters and paths
├── main.py                # training entry point
├── evaluate.py            # evaluation and confusion matrix
├── data/dataset.py        # augmentations and dataloaders
├── models/classifier.py   # efficientnet + classification head
├── training/trainer.py    # train/val loop, checkpointing
├── utils/metrics.py       # sklearn metrics wrappers
├── utils/visualization.py # confusion matrix and curve plots
└── results/               # saved figures
```

## License

MIT
