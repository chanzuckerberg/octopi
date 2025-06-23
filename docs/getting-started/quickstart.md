# Quick Start Guide

This guide walks you through a complete Octopi workflow: from data preparation to particle localization. 

## Basic Workflow Overview

1.	🎯 **Create Training Targets**- Convert particle coordinates (from annotations or portals) into semantic segmentation masks for supervised training.
2.	🧠 **Train a Deep Learning Model** - Train a 3D U-Net model using your masks and tomograms. You can combine multiple datasets and track performance with MLflow.
3.	🔮 **Predict New Segmentations** - Use the trained model to generate segmentation masks for new tomograms. These masks identify potential particle locations.
4.	📍 **Localize particles from masks** - Extract 3D coordinates from the prediction masks.
5.	📊 **Evaluate performance**  - Compare your predicted coordinates against ground truth annotations to calculate metrics like precision, recall, and F1 score.

### Step 1. Prepare Training Labels

Create semantic masks for your proteins of interest using annotation metadata:

```bash
octopi create-targets
    --config config.json
    --tomo-alg wbp --voxel-size 10
    --picks-user-id data-portal --picks-session-id 0
    --target-session-id 1 --target-segmentation-name targets
    --target-user-id octopi
```

🎯 This creates training targets for a single copick query. To produce targets from multiple coordinate queries,  refer to the [Prepare Labels](../user-guide/labels.md) section.

### Step 2. Train a Model

Train a 3D U-Net model:

```bash
octopi train-model
    --config experiment,config1.json
    --config simulation,config2.json
    --voxel-size 10 --tomo-alg wbp --Nclass 8 # Adjust me based on Nclasses present
    --tomo-batch-size 15 --num-epochs 500 --val-interval 10
    --target-info targets,octopi,1
```

**🧪 Note:** `--Nclass` should be the number of distinct object classes + 1 (for background).

We can provide config files stemming from multiple copick projects. This would be relevenant in instances where you want to train a model that reflects multiple experimental acquisitions.

📁 The results will be saved to a `results/` folder which contains the trained model, a config file for the model, and plotted training / validation curves. 

### Step 3. Generate Predicted Segmentation Masks

Apply your trained model to new tomograms:

```bash
octopi segment
    --config config.json
    --seg-info predict,unet,1
    --model-weights results/best_model.pth
    --model-config results/best_model_config.yaml
    --voxel-size 10 --tomo-alg wbp --tomo-batch-size 25
```

This generates segmentation masks for your tomograms provided under the `--voxel-size` and `--tomo-alg` flags. The segmentation masks will be saved under the `--seg-info` flag. 

### Step 4: Extract Particle Coordinates

Convert segmentation masks into precise 3D particle coordinates:

```bash
octopi localize
    --config config.json
    --seg-info predict,unet,1
    --pick-session-id 1 --pick-user-id unet
```

### Step 5: Evaluate the Coordinates

Compare your predicted coordinates with ground truth annotations:

```bash
octopi evaluate 
    --config config.json 
    --ground-truth-user-id data-portal --ground-truth-session-id 0 
    --predict-user-id octopi --predict-session-id 1 
    --save-path analysis
```

## What's Next?

This workflow gives you a quick introduction to the particle picking pipeline. To learn more:

- **[Prepare Labels](../user-guide/labels.md)** - Generate targets for training octopi models.
- **[Train Models](../user-guide/training.md)** - Train a single model or ensemble of models with bayesian optimization.
- **[Inference](../user-guide/inference.md)** - Deploy the trained models and get 3D coordinates.