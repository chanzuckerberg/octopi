# Quick Start Guide

This guide will help you get started with octopi quickly. We'll walk through a basic workflow for training a model and performing inference.

## Prerequisites

- octopi installed (see [Installation Guide](installation.md))
- A CoPick project with tomograms and annotations
- Basic understanding of cryo-ET data

## Basic Workflow

### 1. Data Import

First, import your tomograms into a CoPick project:

```bash
# For local MRC files
octopi import-mrc-volumes \
    --input-folder /path/to/mrc/files \
    --config /path/to/config.json \
    --target-tomo-type denoised \
    --output-voxel-size 10

# Or download from data portal
octopi download-dataportal \
    --config /path/to/config.json \
    --datasetID 10445 \
    --output-voxel-size 10
```

### 2. Create Training Targets

Generate segmentation targets for training:

```bash
octopi create-targets \
    --config config.json \
    --target apoferritin \
    --target ribosome \
    --voxel-size 10 \
    --target-session-id 1 \
    --target-segmentation-name targets
```

### 3. Train a Model

Train a basic 3D U-Net model:

```bash
octopi train-model \
    --config config.json \
    --voxel-size 10 \
    --Nclass 3 \
    --num-epochs 100 \
    --target-info targets,octopi,1
```

### 4. Perform Inference

Generate segmentation predictions:

```bash
octopi inference \
    --config config.json \
    --seg-info predict,unet,1 \
    --model-config results/best_model_config.yaml \
    --model-weights results/best_model.pth \
    --voxel-size 10
```

### 5. Localize Particles

Convert segmentations to particle coordinates:

```bash
octopi localize \
    --config config.json \
    --pick-session-id 1 \
    --pick-user-id unet \
    --seg-info predict,unet,1
```

## Python API Quick Start

Here's a minimal example using the Python API:

```python
from octopi.entry_points.run_create_targets import create_all_train_targets
from octopi.models import common as builder
from octopi.datasets import generators
from octopi.pytorch import trainer

# Create targets
create_all_train_targets(
    config='config.json',
    seg_targets=[],
    picks_session_id=None,
    picks_user_id='data-portal',
    voxel_size=10,
    radius_scale=0.7,
    tomogram_algorithm='wbp',
    target_name='targets',
    target_user_id='octopi'
)

# Train model
data_generator = generators.TrainLoaderManager(
    config='config.json',
    target_name='targets',
    target_user_id='octopi',
    tomo_algorithm='wbp',
    voxel_size=10
)

model = builder.build_model({
    'architecture': 'Unet',
    'channels': [32, 64, 128],
    'strides': [2, 2, 1],
    'num_res_units': 2,
    'num_classes': 3,
    'dropout': 0.1,
    'dim_in': 128
})

trainer = trainer.Trainer(
    model=model,
    data_generator=data_generator,
    model_save_path='results'
)

trainer.train(num_epochs=100)
```

## Next Steps

- Explore the [CLI Tutorial](../user-guide/cli-tutorial.md) for detailed command usage
- Learn about the [Python API](../user-guide/api-tutorial.md) for advanced usage
- Check out [Model Exploration](../user-guide/model-exploration.md) for optimizing your models
- Read about [MLflow Integration](../advanced/mlflow.md) for experiment tracking 