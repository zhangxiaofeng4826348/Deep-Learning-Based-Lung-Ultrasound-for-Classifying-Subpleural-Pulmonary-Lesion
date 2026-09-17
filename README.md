# SLLA-UNet: Deep Learning-Based Lung Ultrasound Decision Support for Subpleural Pulmonary Lesions

This repository contains a cleaned implementation of **SLLA-UNet**, a joint segmentation-classification framework for lung ultrasound analysis of subpleural pulmonary lesions (SPLs). The code is intended to support reproducibility of the model architecture, training workflow, ablation settings, evaluation metrics, and Grad-CAM-based visual outputs described in the manuscript.

## Scope of this public repository

This public release includes:

- SLLA-UNet model definitions and ablation variants
- self-supervised contrastive pretraining code
- supervised fine-tuning code
- evaluation utilities for AUC, accuracy, sensitivity, specificity, precision, F1-score, Dice, and IoU
- Grad-CAM visualization utility
- a logistic-regression baseline utility
- a few de-identified example ultrasound images for demonstrating file format only
- a synthetic toy-data generator for checking that the scripts run

This public release does **not** include the institutional validation datasets, patient-level labels, masks, or clinical variables. Access to institutional data is subject to ethics approval, institutional approval, and data-use agreements.

## Repository structure

```text
SLLA-UNet-assisted-ultrasound-for-SPLs/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── ssl_config.yaml
│   └── finetune_config.yaml
├── slla_unet/
│   ├── __init__.py
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── optimizers.py
│   └── utils.py
├── scripts/
│   ├── clinical_baselines.py
│   ├── create_toy_dataset.py
│   ├── evaluate.py
│   ├── grad_cam.py
│   ├── train_finetune.py
│   └── train_ssl.py
├── examples/
│   └── demo_images/
└── docs/
    ├── dataset_datasheet.md
    └── model_card.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code was prepared for Python 3.8+ and PyTorch. GPU acceleration is recommended for full training.

## Input size

All public scripts use **224 × 224** image input size by default. The image and mask loaders resize inputs to 224 × 224. This is aligned with the supplementary methods description.

## Data format

For supervised fine-tuning and evaluation, provide a CSV with the following columns:

```csv
image,mask,label
train/images/case_001.png,train/masks/case_001_mask.png,1
train/images/case_002.png,train/masks/case_002_mask.png,0
```

Labels: `0 = benign`, `1 = malignant`.

Paths may be absolute or relative to `--root-dir`.

## Quick smoke test with synthetic toy data

```bash
python scripts/create_toy_dataset.py --out-dir examples/toy_data --n-train 8 --n-val 4
python scripts/train_finetune.py \
  --train-csv examples/toy_data/train.csv \
  --val-csv examples/toy_data/val.csv \
  --root-dir examples/toy_data \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --variant non-swin \
  --out-dir outputs/toy_finetune
```

The toy dataset is synthetic and is provided only to test the code path. The CPU smoke-test command uses the non-Swin variant to keep runtime low. Use `--variant full` for the full SLLA-UNet architecture. The toy data are not clinical data and must not be used for scientific evaluation.

## Self-supervised pretraining

```bash
python scripts/train_ssl.py \
  --image-dir /path/to/public_lung_ultrasound_images \
  --epochs 500 \
  --batch-size 12 \
  --input-size 224 \
  --out-dir outputs/ssl
```

The manuscript used a public Figshare lung ultrasound dataset for self-supervised pretraining and institutional SPL cohorts for supervised model development and validation. The pretraining dataset was independent of all SPL cohorts.

## Supervised fine-tuning

```bash
python scripts/train_finetune.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/validation.csv \
  --root-dir /path/to/data_root \
  --ssl-checkpoint outputs/ssl/ssl_best.pth \
  --epochs 100 \
  --batch-size 8 \
  --input-size 224 \
  --class-counts 210 484 \
  --out-dir outputs/finetune
```

The default fine-tuning hyperparameters are:

- encoder learning rate: `5e-6`
- decoder learning rate: `1e-5`
- classifier learning rate: `1e-5`
- weight decay: `1e-5`
- minimum learning rate: `1e-6`
- epochs: `100`
- batch size: `8`
- joint loss weights: `lambda_cls = 1.0`, `lambda_seg = 1.0`

## Ablation variants

Use `--variant` to select ablation settings:

```bash
--variant full          # full SLLA-UNet
--variant non-ssl       # full architecture trained without SSL initialization
--variant non-ms        # no multi-scale feature aggregation
--variant non-swin      # no Swin Transformer branch
--variant non-ms-swin   # no multi-scale aggregation and no Swin branch
```

## Evaluation

```bash
python scripts/evaluate.py \
  --csv /path/to/test.csv \
  --root-dir /path/to/data_root \
  --checkpoint outputs/finetune/best_auc.pth \
  --input-size 224 \
  --threshold 0.5 \
  --out-dir outputs/evaluation
```

The binary classification threshold is `probability >= 0.50` for malignant and `< 0.50` for benign.

## Grad-CAM visualization

```bash
python scripts/grad_cam.py \
  --image examples/demo_images/sample_001.png \
  --checkpoint outputs/finetune/best_auc.pth \
  --target-layer RS \
  --out outputs/grad_cam_sample_001.png
```

## Baseline models

The manuscript included two logistic-regression reference models: a clinical model and an ultrasound feature-based model. This repository includes a generic logistic-regression utility:

```bash
python scripts/clinical_baselines.py \
  --train-csv /path/to/baseline_train_features.csv \
  --test-csv /path/to/baseline_test_features.csv \
  --label-col label \
  --out-dir outputs/baseline
```

The input CSV should contain one label column and numeric feature columns. Preprocessing and feature definitions should follow the study protocol.

## Data availability

The institutional SPL datasets are not included in this public repository because public release of patient-level images and labels has not been approved by the relevant institutional and ethical review authorities. De-identified data may be available from the corresponding author on reasonable request, subject to institutional approval, IRB approval where required, and a signed data-use agreement.

## License

The source code is released under the MIT License. The MIT License applies to code only and does not grant rights to any patient image data.
