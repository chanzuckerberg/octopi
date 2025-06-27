# Octopi 🐙🐙🐙 API

## Introduction

Octopi is a powerful framework for **3D CNN instance segmentation of proteins in Cryo-ET tomograms**. Built on top of MONAI and PyTorch, octopi provides a complete pipeline for training deep learning models to identify and segment macromolecular structures in cryo-electron tomography data.

### Key Features

- **🔧 Flexible Configuration**: Modular design supporting various model architectures and training strategies  
- **📊 Comprehensive Metrics**: Built-in evaluation with confusion matrices, F1 scores, precision, and recall
- **⚡ Memory Efficient**: Smart data loading for large tomogram datasets that don't fit in memory
- **🎯 Specialized Loss Functions**: Weighted Focal Tversky Loss optimized for class-imbalanced volumetric data
- **🔍 Hyperparameter Optimization**: Integrated Optuna support for automated model exploration
- **📈 Experiment Tracking**: MLflow integration for reproducible research