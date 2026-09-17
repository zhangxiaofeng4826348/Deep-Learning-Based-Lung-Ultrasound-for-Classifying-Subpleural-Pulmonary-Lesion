# Dataset Datasheet Summary

## Public repository data

This repository includes only a small number of de-identified example ultrasound images for demonstrating file format and a script for generating synthetic toy data. These examples are not intended for model evaluation.

## Institutional study data

The institutional SPL datasets used in the manuscript are not included in this repository. Public release of patient-level images, labels, masks, and clinical variables has not been approved by the relevant institutional and ethical review authorities.

## Expected supervised data schema

A supervised dataset CSV should contain:

- `image`: path to B-mode ultrasound image
- `mask`: path to binary lesion mask
- `label`: 0 for benign, 1 for malignant

## Data governance

Data access to institutional SPL datasets may be considered on reasonable request to the corresponding author and is subject to institutional approval, IRB approval where required, and a signed data-use agreement.

## Pretraining data

The manuscript used an independent public lung ultrasound dataset for self-supervised pretraining. Users should cite and comply with the license/terms of the source dataset when reproducing pretraining.
