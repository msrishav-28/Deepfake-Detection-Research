# Results Directory

This directory contains evaluation results from the deepfake detection research.

## Structure

```
results/
├── evaluation/           # Model evaluation results
│   ├── model_comparison.csv
│   ├── detailed_results.json
│   └── explainability/  # Grad-CAM visualizations
└── training/            # Training logs and metrics
```

## Deepfake Detection Results

This folder will contain the results from your deepfake detection experiments:

### ImageNet Validation - [`results-imagenet.csv`](results-imagenet.csv)

The standard 50,000 image ImageNet-1k validation set. Model selection during training utilizes this validation set, so it is not a true test set. Question: Does anyone have the official ImageNet-1k test set classification labels now that challenges are done?

* Source: http://image-net.org/challenges/LSVRC/2012/index
* Paper: "ImageNet Large Scale Visual Recognition Challenge" - https://arxiv.org/abs/1409.0575

### ImageNet-"Real Labels" - [`results-imagenet-real.csv`](results-imagenet-real.csv)

The usual ImageNet-1k validation set with a fresh new set of labels intended to improve on mistakes in the original annotation process.

* Source: https://github.com/google-research/reassessed-imagenet
* Paper: "Are we done with ImageNet?" - https://arxiv.org/abs/2006.07159

### ImageNetV2 Matched Frequency - [`results-imagenetv2-matched-frequency.csv`](results-imagenetv2-matched-frequency.csv)

An ImageNet test set of 10,000 images sampled from new images roughly 10 years after the original. Care was taken to replicate the original ImageNet curation/sampling process.

* Source: https://github.com/modestyachts/ImageNetV2
* Paper: "Do ImageNet Classifiers Generalize to ImageNet?" - https://arxiv.org/abs/1902.10811

### ImageNet-Sketch - [`results-sketch.csv`](results-sketch.csv)

50,000 non photographic (or photos of such) images (sketches, doodles, mostly monochromatic) covering all 1000 ImageNet classes.

* Source: https://github.com/HaohanWang/ImageNet-Sketch
* Paper: "Learning Robust Global Representations by Penalizing Local Predictive Power" - https://arxiv.org/abs/1905.13549

### ImageNet-Adversarial - [`results-imagenet-a.csv`](results-imagenet-a.csv)

A collection of 7500 images covering 200 of the 1000 ImageNet classes. Images are naturally occurring adversarial examples that confuse typical ImageNet classifiers. This is a challenging dataset, your typical ResNet-50 will score 0% top-1.

For clean validation with same 200 classes, see [`results-imagenet-a-clean.csv`](results-imagenet-a-clean.csv) 

* Source: https://github.com/hendrycks/natural-adv-examples
* Paper: "Natural Adversarial Examples" - https://arxiv.org/abs/1907.07174

### ImageNet-Rendition - [`results-imagenet-r.csv`](results-imagenet-r.csv)

Renditions of 200 ImageNet classes resulting in 30,000 images for testing robustness.

For clean validation with same 200 classes, see [`results-imagenet-r-clean.csv`](results-imagenet-r-clean.csv) 

* Source: https://github.com/hendrycks/imagenet-r
* Paper: "The Many Faces of Robustness" - https://arxiv.org/abs/2006.16241

### TODO
* Explore adding a reduced version of ImageNet-C (Corruptions) and ImageNet-P (Perturbations) from https://github.com/hendrycks/robustness. The originals are huge and image size specific.


## Benchmark

CSV files with a `model_benchmark` prefix include benchmark numbers for models on various accelerators with different precision. Currently only run on RTX 3090 w/ AMP for inference, I intend to add more in the future.

## Metadata

CSV files with `model_metadata` prefix contain extra information about the source training, currently the pretraining dataset and technique (ie distillation, SSL, WSL, etc). Eventually I'd like to have metadata about augmentation, regularization, etc. but that will be a challenge to source consistently. 
