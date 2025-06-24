# API Tutorial

This guide demonstrates how to use octopi's Python API for training and inference tasks. We'll walk through the process of training a 3D U-Net model for instance segmentation of proteins in Cryo-ET tomograms.

## Data Preparation

### Generating Training Targets

First, we need to prepare the target data for training. This involves creating spherical targets corresponding to protein locations in the tomograms.

```python
from octopi.entry_points.run_create_targets import create_sub_train_targets, create_all_train_targets

# Configuration
config = '10440_config.json'
target_name = 'targets'
target_user_id = 'octopi'
target_session_id = '0'
voxel_size = 10.012
tomogram_algorithm = 'wbp-denoised-denoiset-ctfdeconv'
radius_scale = 0.7

# Option 1: Specify specific pickable objects
pick_targets = [
    ('ribosome', 'data-portal', None),
    ('virus-like-particle', 'data-portal', None),
    ('apoferritin', 'data-portal', None)
]

seg_targets = []

create_sub_train_targets(
    config, pick_targets, seg_targets, voxel_size, radius_scale, tomogram_algorithm,
    target_name, target_user_id, target_session_id
)

# Option 2: Use all available pickable objects
picks_user_id = 'data-portal'
picks_session_id = None

create_all_train_targets(
    config, seg_targets, picks_session_id, picks_user_id, 
    voxel_size, radius_scale, tomogram_algorithm, 
    target_name, target_user_id, target_session_id
)
```

## Model Training

### Setting Up the Training Environment

```python
from monai.metrics import ConfusionMatrixMetric
from octopi.models import common as builder
from octopi.datasets import generators
from monai.losses import TverskyLoss
from octopi import losses
from octopi.pytorch import trainer 
from octopi import io, utils
import torch, os

# Training Parameters
config = "10440_config.json"
target_name = 'targets'
target_user_id = 'octopi'
target_session_id = None

# Data Generator Parameters
num_tomo_crops = 16
tomo_algorithm = 'wbp-denoised-denoiset-ctfdeconv'
voxel_size = 10.012
tomo_batch_size = 25

# Model Configuration
Nclass = 7
model_config = {
    'architecture': 'Unet',
    'channels': [32, 64, 128, 128],
    'strides': [2, 2, 1, 1],
    'num_res_units': 3,
    'num_classes': Nclass,
    'dropout': 0.05,
    'dim_in': 128
}

model_save_path = 'results'
model_weights = None
```

### Creating the Data Generator

```python
data_generator = generators.TrainLoaderManager(
    config, 
    target_name, 
    target_session_id=target_session_id,
    target_user_id=target_user_id,
    tomo_algorithm=tomo_algorithm,
    voxel_size=voxel_size,
    num_tomo_crops=num_tomo_crops,
    tomo_batch_size=tomo_batch_size
)
```

### Training the Model

```python
# Initialize the model
model = builder.build_model(model_config)

# Define loss function and metrics
loss_fn = losses.WeightedFocalTverskyLoss()
metric_fn = ConfusionMatrixMetric(include_background=True)

# Create trainer
trainer = trainer.Trainer(
    model=model,
    data_generator=data_generator,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    model_save_path=model_save_path,
    model_weights=model_weights
)

# Start training
trainer.train(
    num_epochs=100,
    val_interval=10
)
```

## Model Exploration with Optuna

For automatic model architecture search:

```python
from octopi.optimization import optuna_optimizer

optimizer = optuna_optimizer.OptunaOptimizer(
    config=config,
    target_name=target_name,
    target_user_id=target_user_id,
    tomo_algorithm=tomo_algorithm,
    voxel_size=voxel_size,
    model_save_path='train_results'
)

optimizer.optimize(
    n_trials=50,
    timeout=3600  # 1 hour timeout
)
```

## Best Practices

1. **Data Resolution**: Tomograms should be resampled to at least 10 Å per voxel for optimal performance and memory usage.

2. **Memory Management**: Use the `tomo_batch_size` parameter to control memory usage when training on large datasets.

3. **Model Architecture**: Start with the default U-Net configuration and use Optuna for architecture optimization if needed.

4. **Data Augmentation**: The data generator includes built-in augmentation techniques suitable for cryo-ET data.

## Next Steps

- For inference examples, see the [Inference Guide](inference.md)
- For detailed API reference, see the [API Documentation](../api/core.md)
- For advanced usage, see the [Advanced Topics](../advanced/mlflow.md) section 