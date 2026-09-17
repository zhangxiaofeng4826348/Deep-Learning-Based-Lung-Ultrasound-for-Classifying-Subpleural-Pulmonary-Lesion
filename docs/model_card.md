# Model Card: SLLA-UNet

## Model details

SLLA-UNet is a deep learning-based lung ultrasound decision-support model for analysis of subpleural pulmonary lesions. The public code defines a joint segmentation-classification architecture integrating a U-Net-like convolutional branch, multi-scale encoder feature aggregation, and an optional Swin Transformer branch.

## Intended use

The model is intended for research use as an adjunctive decision-support tool for SPL benign-malignant classification on B-mode lung ultrasound images. It is not intended for autonomous diagnosis or replacement of histopathologic or clinical decision-making.

## Input

- B-mode lung ultrasound image
- RGB tensor resized to 224 × 224

## Output

- lesion segmentation logits
- malignancy probability from the classification branch
- intermediate feature maps usable for Grad-CAM visualization

## Training workflow

- Self-supervised contrastive pretraining using external unlabeled lung ultrasound images
- Supervised fine-tuning with paired SPL images, segmentation masks, and benign/malignant labels

## Limitations

- Public release does not include institutional patient-level image data
- Performance estimates depend on dataset composition, scanner distribution, image-selection procedure, and local disease prevalence
- Calibration should be assessed locally before deployment
- Visual explanation outputs are associative and do not constitute causal explanations of model reasoning

## Monitoring considerations

Before clinical deployment, local validation should examine discrimination, calibration, subgroup performance, failure modes, and clinician interaction effects.
