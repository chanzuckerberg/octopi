# CLI Tutorial

This guide will walk you through using octopi's command-line interface for various tasks in the particle picking pipeline.

## Basic Usage

octopi provides a clean, scriptable command-line interface. Run the following command to view all available subcommands:

```bash
octopi --help
```

Each subcommand supports its own `--help` flag for detailed usage.

## Data Import & Preprocessing

### Importing Local MRC Files

If your tomograms are already processed and stored locally in .mrc format (e.g., from Warp, IMOD, or AreTomo), you can import them into a new or existing CoPick project using:

```bash
octopi import-mrc-volumes \
    --input-folder /path/to/mrc/files \
    --config /path/to/config.json \
    --target-tomo-type denoised \
    --input-voxel-size \
    --output-voxel-size 10
```

### Downloading from Data Portal

To download and process tomograms from the data portal:

```bash
octopi download-dataportal \
    --config /path/to/config.json \
    --datasetID 10445 \
    --overlay-path path/to/saved/zarrs \
    --input-voxel-size 5 \
    --output-voxel-size 10 \
    --dataportal-name wbp \
    --target-tomotype wbp
```

## Training Labels Preparation

Generate semantic masks for proteins of interest using annotation metadata:

```bash
octopi create-targets \
    --config config.json \
    --target apoferritin \
    --target beta-galactosidase,slabpick,1 \
    --target ribosome,pytom,0 \
    --target virus-like-particle,pytom,0 \
    --seg-target membrane \
    --tomo-alg wbp \
    --voxel-size 10 \
    --target-session-id 1 \
    --target-segmentation-name remotetargets \
    --target-user-id train-octopi
```

## Training a Model

Train a 3D U-Net model on prepared datasets:

```bash
octopi train-model \
    --config experiment,config1.json \
    --config simulation,config2.json \
    --voxel-size 10 \
    --tomo-alg wbp \
    --Nclass 8 \
    --tomo-batch-size 50 \
    --num-epochs 100 \
    --val-interval 10 \
    --target-info remotetargets,train-octopi,1
```

## Model Exploration

Launch a model exploration job using Optuna:

```bash
octopi model-explore \
    --config experiment,/mnt/dataportal/ml_challenge/config.json \
    --config simulation,/mnt/dataportal/synthetic_ml_challenge/config.json \
    --voxel-size 10 \
    --tomo-alg wbp \
    --Nclass 8 \
    --model-save-path train_results
```

## Inference

Generate segmentation prediction masks:

```bash
octopi inference \
    --config config.json \
    --seg-info predict,unet,1 \
    --model-config train_results/best_model_config.yaml \
    --model-weights train_results/best_model.pth \
    --voxel-size 10 \
    --tomo-alg wbp \
    --tomo-batch-size 25
```

## Localization

Convert segmentation masks into particle coordinates:

```bash
octopi localize \
    --config config.json \
    --pick-session-id 1 \
    --pick-user-id unet \
    --seg-info predict,unet,1
```

## HPC Cluster Usage

If you're running octopi on an HPC cluster, several SLURM-compatible submission commands are available:

```bash
octopi-slurm --help
```

This provides utilities for submitting training, inference, and localization jobs in SLURM-based environments. 