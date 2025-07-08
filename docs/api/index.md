# Octopi API Documentation

Octopi is a comprehensive 3D particle picking framework designed for cryo-electron tomography data analysis. This documentation covers the complete workflow from training to inference and evaluation.

## Overview

Octopi provides a streamlined pipeline for:

- **Training**: Deep learning models for particle segmentation
- **Inference**: Automated particle detection and localization
- **Evaluation**: Performance assessment against ground truth annotations

## Quick Start

For a minimal introduction to all core functions with essential parameters, see the [Quick Start Guide](quick-start.md). The sections below describe each component in greater detail.

## Core Components

### Configuration
All octopi workflows start with a Copick configuration file that defines:

- Data locations and formats
- Pickable object definitions with corresponding segmentation label values
- Tomogram metadata and processing parameters

The configuration file maps each pickable object to a specific integer value used in segmentation masks, enabling multi-class particle detection and classification.

## Workflow Pages

### [Training](training.md)
Learn how to:

- Create training targets from existing annotations
- Configure and train deep learning models
- Set up cross-validation splits
- Choose appropriate loss functions and model architectures

### [Inference](inference.md)
Discover how to:

- Run segmentation on new tomograms
- Perform particle localization from segmentation masks
- Configure test-time augmentation
- Evaluate results against ground truth