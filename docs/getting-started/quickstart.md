# Quick Start Guide

This guide walks you through a complete OCTOPI workflow: from data preparation to particle localization. 

## Basic Workflow

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

This creates training targets for a single copick query. To produce targets from multiple coordinate queries,  refer to the [Prepare Labels](../user-guide/labels.md) section.

### Step 2. Train a Model

Train a 3D U-Net model:

```bash
octopi train-model
    --config experiment,config1.json
    --config simulation,config2.json
    --voxel-size 10 --tomo-alg wbp --Nclass 8
    --tomo-batch-size 50 --num-epochs 100 --val-interval 10
    --target-info targets,octopi,1
```

We can provide config files stemming from multiple copick projects. This would be relevenant in instances where you want to train a model that reflects multiple experimental acquisitions. For the number of classes, this value is a number of pickable objects + 1 to account for background. 

The results will be saved to a `results/` folder which contains the trained model, a config file for the model, and plotted training / validation curves. 

### Step 3. Generate Predicted Segmentation Masks

Apply your trained model to new tomograms:

```bash
octopi inference
    --config config.json
    --seg-info predict,unet,1
    --model-weights results/best_model.pth
    --model-config results/best_model_config.yaml
    --voxel-size 10 --tomo-alg wbp --tomo-batch-size 25
```

This generates segmentation masks for your tomograms provided under the `--voxel-size` and `--tomo-alg` flags. The segmentation masks will be saved under the `--seg-info` flag. 

### Step 4: Extract Particle Coordinates

Convert segmentation masks into precise particle coordinates:

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

This workflow gives you a quick particle picking pipeline. To learn more:

- **[Prepare Labels](../user-guide/labels.md)**
- **[Training Details](../user-guide/training.md)** - Explore model exploration
- **[Inference](../user-guide/inference.md)** - Learn about different inference modes and post-processing
