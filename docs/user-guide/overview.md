# User Guide Overview

Welcome to the Octopi User Guide! This comprehensive tutorial series will take you from raw tomogram data to precise particle coordinates using deep learning-based 3D particle picking.

## Tutorial Sections

### 🏷️ [Prepare Labels](labels.md)
**Create training targets from particle annotations**

Learn how to convert particle coordinates into 3D training masks. This section covers:

- Automated target generation from data portal annotations
- Manual specification for custom datasets
- Quality control and validation techniques
- Multi-class segmentation preparation

**When to use:** Start here after importing your data to prepare training materials.

### 🧠 [Training](training.md)
**Train 3D U-Net models with Bayesian optimization**

Master both single model training and automated architecture exploration:

- Model exploration with Bayesian optimization (recommended)
- Single model training for specific use cases
- MLflow experiment tracking and monitoring
- Best practices for resource management

**When to use:** After preparing training labels, use this to develop your particle picking models.

### 🔮 [Inference and Localization](inference.md)
**Apply trained models to generate predictions and extract particle coordinates**

Deploy your trained models to analyze new tomograms:
- Segmentation mask generation
- Peak detection and particle extraction
- Performance evaluation against ground truth

**When to use:** Once you have trained models, use this to get final particle coordinates.

## Choose Your Path

### 🚀 **New to OCTOPI?**
Follow the complete workflow:
1. [Prepare Labels](labels.md) → [Training](training.md) → [Inference](inference.md)

### 🔬 **Have existing models?**
Skip training and go directly to:
- [Inference](inference.md) → [Localization](localization.md)

### 💻 **Python developer?**
Check out the [API Tutorial](api-tutorial.md) for programmatic usage

## Key Concepts

### Copick Integration
OCTOPI uses [copick](https://github.com/copick/copick) for data management, providing seamless access to tomograms and annotations across local and remote storage.

### Bayesian Optimization
Model exploration uses Bayesian methods to efficiently search the space of possible architectures and hyperparameters.

## What's Next?

Ready to start? Choose your entry point:

- **[Begin with Labels →](labels.md)** - Start the complete workflow
- **[Jump to Training →](training.md)** - If you already have training targets
- **[Skip to Inference →](inference.md)** - If you have pre-trained models
- **[Explore the API →](api-tutorial.md)** - For programmatic usage

---

*Each tutorial builds on the previous ones, but you can jump to specific sections based on your needs and existing progress.*