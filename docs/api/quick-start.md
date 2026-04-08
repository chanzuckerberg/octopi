# Quick Start

This page provides a minimal introduction to all core octopi functions to get you up and running quickly. For detailed explanations and advanced options, see the [Training](training.md) and [Inference](inference.md) pages.

## Prerequisites

- CoPick configuration file pointing to your tomogram data
- Existing particle annotations (picks) or segmentations for training
- Python environment with octopi installed

## Complete Workflow

Here's the essential 5-step workflow from data preparation to evaluation:

### 1. Create Training Targets

```python
from octopi.entry_points.run_create_targets import create_sub_train_targets

config = 'config.json'
voxel_size = 10.012
tomo_algorithm = 'denoised'
radius_scale = 0.7

# Define source annotations as (object_name, user_id, session_id)
pick_targets = [
    ('ribosome', 'data-portal', '1'),
    ('virus-like-particle', 'data-portal', '1'),
    ('apoferritin', 'data-portal', '1')
]

create_sub_train_targets(
    config, pick_targets, [], voxel_size, radius_scale,
    tomo_algorithm, 'targets', 'octopi', '1'
)
```

!!! tip
    Browse available data with `copick browse -c config.json` (local) or `copick browse -ds <datasetID>` (Data Portal).

### 2. Train Model

```python
from octopi.datasets.config import DataGeneratorConfig
from octopi.workflows import train
from monai.losses import TverskyLoss

config = 'config.json'
results_folder = 'model_output'

# Build the data generator
cfg = DataGeneratorConfig(
    config=config,
    name='targets', user_id='octopi', session_id='1',
    voxel_size=10.012, tomo_algorithm='denoised',
)
data_generator = cfg.create_data_generator()

# Model architecture
model_config = {
    'architecture': 'Unet',
    'num_classes': data_generator.Nclasses,  # objects + background
    'dim_in': 80,
    'strides': [2, 2, 1],
    'channels': [48, 64, 80, 80],
    'dropout': 0.0,
    'num_res_units': 1,
}

# Loss function
loss_function = TverskyLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    alpha=0.3,
    beta=0.7
)

# Train
train(
    data_generator, loss_function,
    model_config=model_config,
    model_save_path=results_folder
)
```

### 3. Run Segmentation

```python
from octopi.workflows import segment

segment(
    config=config,
    tomo_algorithm='denoised',
    voxel_size=10.012,
    model_weights=f'{results_folder}/best_model.pth',
    model_config=f'{results_folder}/model_config.yaml',
    seg_info=['predict', 'octopi', '1'],
    ntta=4  # number of test-time augmentation rotations
)
```

### 4. Extract Coordinates

```python
from octopi.workflows import localize

localize(
    config=config,
    voxel_size=10.012,
    seg_info=['predict', 'octopi', '1'],
    pick_user_id='octopi',
    pick_session_id='1'
)
```

### 5. Evaluate Results

```python
from octopi.workflows import evaluate

evaluate(
    config=config,
    gt_user_id='data-portal',
    gt_session_id='1',
    pred_user_id='octopi',
    pred_session_id='1',
    distance_threshold=0.5,
    save_path=f'{results_folder}/evaluation'
)
```

??? tip "All in one script"

    Copy and paste this complete script, then modify the configuration variables at the top:

    ```python
    from octopi.entry_points.run_create_targets import create_sub_train_targets
    from octopi.datasets.config import DataGeneratorConfig
    from octopi.workflows import train, segment, localize, evaluate
    from monai.losses import TverskyLoss

    # =============================================================================
    # CONFIGURATION - Modify these variables for your dataset
    # =============================================================================

    config = 'config.json'
    voxel_size = 10.012
    tomo_algorithm = 'denoised'
    results_folder = 'model_output'

    pick_targets = [
        ('ribosome', 'data-portal', '1'),
        ('virus-like-particle', 'data-portal', '1'),
        ('apoferritin', 'data-portal', '1')
    ]

    gt_user_id = 'data-portal'
    gt_session_id = '1'

    # =============================================================================
    # WORKFLOW
    # =============================================================================

    print("Step 1: Creating training targets...")
    create_sub_train_targets(
        config, pick_targets, [], voxel_size, 0.7,
        tomo_algorithm, 'targets', 'octopi', '1'
    )

    print("Step 2: Training model...")
    cfg = DataGeneratorConfig(
        config=config,
        name='targets', user_id='octopi', session_id='1',
        voxel_size=voxel_size, tomo_algorithm=tomo_algorithm,
    )
    data_generator = cfg.create_data_generator()

    model_config = {
        'architecture': 'Unet',
        'num_classes': data_generator.Nclasses,
        'dim_in': 80,
        'strides': [2, 2, 1],
        'channels': [48, 64, 80, 80],
        'dropout': 0.0,
        'num_res_units': 1,
    }

    loss_function = TverskyLoss(
        include_background=True, to_onehot_y=True, softmax=True,
        alpha=0.3, beta=0.7
    )

    train(
        data_generator, loss_function,
        model_config=model_config,
        model_save_path=results_folder
    )

    print("Step 3: Running segmentation...")
    segment(
        config=config,
        tomo_algorithm=tomo_algorithm,
        voxel_size=voxel_size,
        model_weights=f'{results_folder}/best_model.pth',
        model_config=f'{results_folder}/model_config.yaml',
        seg_info=['predict', 'octopi', '1'],
        ntta=4
    )

    print("Step 4: Extracting coordinates...")
    localize(
        config=config,
        voxel_size=voxel_size,
        seg_info=['predict', 'octopi', '1'],
        pick_user_id='octopi',
        pick_session_id='1'
    )

    print("Step 5: Evaluating results...")
    evaluate(
        config=config,
        gt_user_id=gt_user_id,
        gt_session_id=gt_session_id,
        pred_user_id='octopi',
        pred_session_id='1',
        distance_threshold=0.5,
        save_path=f'{results_folder}/evaluation'
    )

    print(f"Complete! Results saved to: {results_folder}")
    ```

## Key Parameters

- **`config`**: Path to your CoPick configuration file
- **`voxel_size`**: Tomogram resolution in Angstroms
- **`tomo_algorithm`**: Algorithm identifier used during reconstruction
- **`pick_targets`**: List of `(object_name, user_id, session_id)` for your annotations
- **`num_classes`**: `data_generator.Nclasses` — automatically computed from your targets

## Next Steps

- **[Training Guide](training.md)** — Loss functions, cross-validation, and model exploration
- **[Inference Guide](inference.md)** — Segmentation, localization, and evaluation in detail
- **[Adding New Models](adding-new-models.md)** — Integrate custom architectures
