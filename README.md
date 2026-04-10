# histopathologic-cancer-detection

CNN-based binary classifier for histopathologic cancer detection on the PatchCamelyon (PCam) benchmark. Trained on 220k H&E-stained lymph node patches; achieves **93.22% test accuracy** and **ROC-AUC 0.9805** with a custom PyTorch pipeline including Bayesian hyperparameter optimization.

---

## Table of contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Requirements](#requirements)
- [Authors](#authors)

---

## Overview

This project addresses binary patch-level classification of histopathologic images: given a 96×96 RGB tile extracted from a digitized sentinel lymph node whole-slide image (WSI), the model predicts whether the central 32×32 region contains tumor tissue.

The pipeline covers the full ML lifecycle:

- custom `Dataset` class with on-demand disk loading for large-scale data
- exploratory data analysis and class distribution inspection
- modular `CancerCNN` architecture with configurable depth, filter counts, and dropout
- controlled ablation experiments isolating the effect of augmentation, class balancing, spatial context, and color information
- Bayesian optimization of eight hyperparameters via Gaussian Process surrogate and Expected Improvement acquisition

---

## Dataset

| Split | Images | Positive (cancer) | Negative |
|-------|--------|-------------------|----------|
| Train | ~140,800 | ~57,000 (40.5%) | ~83,800 (59.5%) |
| Validation | ~35,200 | — | — |
| Held-out test | ~44,000 | ~17,800 | ~26,200 |

Source: [Kaggle — Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection), derived from the [Camelyon16 Challenge](https://camelyon16.grand-challenge.org/).

- Image format: `.tif`, RGB, 96×96 px
- Labels: binary (0 = no tumor in center, 1 = tumor present in central 32×32 region)
- Digitized at 40× objective, undersampled to 10×; ground truth by expert pathologists with IHC validation
- No duplicate entries confirmed (220,025 unique IDs)

The `HistopathologicCancerDataset` class stores image IDs and labels in memory and loads images on demand from disk, keeping RAM usage tractable at this scale.

---

## Architecture

`CancerCNN` is a configurable convolutional neural network inheriting from `nn.Module`.

```
Input (3 × 96 × 96)
  └─ Conv block 1: Conv2d(3→16, 3×3) → BN → ReLU → MaxPool(2×2)
  └─ Conv block 2: Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)
  └─ Conv block 3: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
  └─ Conv block 4: Conv2d(64→96, 3×3) → BN → ReLU → MaxPool(2×2)
  └─ Flatten
  └─ FC(256) → ReLU → Dropout(0.5)
  └─ Output(2)  [logits]
```

| Parameter | Baseline value |
|-----------|---------------|
| Conv blocks | 4 |
| Filters | 16 → 32 → 64 → 96 |
| Kernel size | 3×3 |
| Pooling | MaxPool 2×2 |
| FC units | [256] |
| Dropout | 0.5 |
| Activation | ReLU (after BN) |
| Optimizer | Adam, lr=1e-3 |
| Loss | Cross-entropy |
| Epochs | 30 |

The flattened feature size is computed automatically via a dummy forward pass, making the architecture adaptable to different input resolutions.

---

## Experiments

Four controlled ablations, each modifying exactly one aspect of the baseline pipeline:

| ID | Change | Key finding |
|----|--------|-------------|
| Baseline | Reference | AUC 0.9805, high precision, moderate recall |
| Exp1 — Oversampling | WeightedRandomSampler for class balance | Recall ↑ (0.92), FN ↓ 42%; FP cost ×3 |
| Exp2 — NoAug | Remove data augmentation | Best accuracy (0.9463) and AUC (0.9851); ~2× faster training |
| Exp3 — CenterCrop | Input reduced from 96×96 to 32×32 | Severe degradation: AUC 0.8924, FN ×1.8 |
| Exp4 — Grayscale | Single-channel input | Moderate degradation: AUC 0.9694; color cues matter |

Training augmentation (baseline): random horizontal/vertical flip, rotation ±20°, color jitter, affine scaling. Validation and test sets use only tensor conversion and normalization (mean=0.5, std=0.5).

---

## Hyperparameter tuning

Bayesian optimization (BO) over 8 hyperparameters using a Gaussian Process surrogate (Matérn kernel, ν=2.5) and Expected Improvement (EI) acquisition function. Search initialized with 5 Latin Hypercube samples followed by 10 BO iterations.

**Search space:**

| Hyperparameter | Lower | Upper |
|----------------|-------|-------|
| n_filters_1 | 32 | 64 |
| n_filters_2 | 64 | 128 |
| n_filters_3 | 128 | 256 |
| dense_units_1 | 128 | 512 |
| dense_units_2 | 64 | 256 |
| dropout_rate | 0.1 | 0.5 |
| learning_rate | 1e-4 | 1e-2 |
| batch_size | 16 | 64 |

**Best configuration found:**

| Hyperparameter | Value |
|----------------|-------|
| n_filters | 40 / 100 / 200 |
| dense_units | 400 / 90 |
| dropout_rate | 0.20 |
| learning_rate | 7×10⁻⁴ |
| batch_size | 64 |

→ Validation accuracy at best epoch: **94.13%** (epoch 28, train/val loss: 0.2137 / 0.1614)

Key observation: training stability is dominated by learning rate and dropout. Configurations with lr ≈ 9×10⁻³ and dropout ≈ 0.39 collapsed to majority-class prediction (~60% accuracy throughout all 30 epochs).

---

## Results

### Baseline — held-out test set

| Metric | Value |
|--------|-------|
| Accuracy | 93.22% |
| ROC-AUC | 0.9805 |
| Precision (cancer) | 96.67% |
| Recall (cancer) | 86.23% |
| Specificity | 97.97% |
| F1 (cancer) | 0.9115 |
| False positives | 530 |
| False negatives | 2,455 |

### All configurations — summary

| Model | Accuracy | ROC-AUC | Recall | F1 |
|-------|----------|---------|--------|----|
| Baseline | 0.9322 | 0.9805 | 0.8623 | 0.9115 |
| Exp1 (OverS) | 0.9307 | 0.9782 | 0.9204 | 0.9150 |
| Exp2 (NoAug) | **0.9463** | **0.9851** | 0.9193 | **0.9327** |
| Exp3 (Center) | 0.8255 | 0.8924 | 0.7517 | 0.7772 |
| Exp4 (Gray) | 0.9156 | 0.9694 | 0.8913 | 0.8954 |
| BO-tuned | — | — | — | val acc 94.13% |

> **Clinical note.** In a screening context, recall (sensitivity) is the most critical metric — a false negative (missed cancer) carries higher clinical risk than a false positive. Exp2 provides the best aggregate trade-off; Exp1 maximizes recall at the cost of more false positives. Decision threshold calibration on a clinically representative validation set is recommended before any deployment.

---

## Repository structure

```
histopathologic-cancer-detection/
│
├── data/
│   └── README.md                  # instructions for downloading the Kaggle dataset
│
├── src/
│   ├── dataset.py                 # HistopathologicCancerDataset
│   ├── model.py                   # CancerCNN
│   ├── trainer.py                 # CancerTrainer
│   ├── evaluator.py               # CancerEvaluator (metrics, confusion matrix, ROC)
│   ├── experiments.py             # Exp1–Exp4 ablation runners
│   └── bayesian_optimizer.py      # BayesianOptimizationCNN
│
├── notebooks/
│   ├── 01_eda.ipynb               # exploratory data analysis
│   ├── 02_baseline.ipynb          # baseline training and evaluation
│   ├── 03_experiments.ipynb       # ablation study
│   └── 04_hyperparameter_tuning.ipynb
│
├── outputs/
│   ├── checkpoints/               # saved model weights (.pt)
│   ├── figures/                   # confusion matrices, ROC curves, training curves
│   └── logs/                      # training logs per experiment
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/histopathologic-cancer-detection.git
cd histopathologic-cancer-detection
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection) and place it under `data/` with the following structure:

```
data/
├── train/          # .tif image files
├── test/           # .tif image files
└── train_labels.csv
```

---

## Usage

**Train baseline:**
```python
from src.dataset import HistopathologicCancerDataset
from src.model import CancerCNN
from src.trainer import CancerTrainer

trainer = CancerTrainer(data_root="data/", num_epochs=30)
trainer.train()
```

**Run evaluation:**
```python
from src.evaluator import CancerEvaluator

evaluator = CancerEvaluator(model, train_loader, val_loader, test_loader)
evaluator.evaluate_all()
```

**Run Bayesian optimization:**
```python
from src.bayesian_optimizer import BayesianOptimizationCNN

optimizer = BayesianOptimizationCNN(trainer, n_initial=5, n_iterations=10)
best_params, best_val_acc = optimizer.optimize()
```


---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy
pandas
Pillow
scikit-learn
scipy
matplotlib
```

---

## Authors

Carrieri Simone, Serra Tomás, Banfi Marco
*Deep Learning Methods for Biomedicine — Universitat Politècnica de Catalunya (UPC), A.Y. 2025–2026*