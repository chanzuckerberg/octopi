# Quick Start

This page provides a minimal introduction to all core octopi functions to get you up and running quickly. For detailed explanations and advanced options, see the [Training](training.md) and [Inference](inference.md) pages.

## Prerequisites

- Copick configuration file pointing to your tomogram data
- Existing particle annotations (picks) or segmentations for training
- Python environment with octopi installed

## Complete Workflow

Here's the essential 5-step workflow from data preparation to evaluation:

### 1. Create Training Targets

```python
from octopi.entry_points.run_create_targets import create_sub_train_targets

# Configuration
config = 'config.json'
target_name = 'targets'
target_user_id = 'octopi'
target_session_id = '1'

# Tomogram parameters
voxel_size = 10.012
tomo_algorithm = 'denoised'
radius_scale = 0.7

# Define source annotations
pick_targets = [
    ('ribosome', 'data-portal', '1'),
    ('virus-like-particle', 'data-portal', '1'),
    ('apoferritin', 'data-portal', '1')
]

# Create training targets
create_sub_train_targets(
    config, pick_targets, [], voxel_size, radius_scale,
    tomo_algorithm, target_name, target_user_id, target_session_id
)
```
🔬 Check available data with: <code>copick browse -c config.json

### 2. Train Model

```python
from octopi.workflows import train
from monai.losses import TverskyLoss

# Training configuration
target_info = ['targets', 'octopi', '1']
results_folder = 'model_output'

# Model architecture
model_config = {
    'architecture': 'Unet',
    'num_classes': 4,  # 3 objects + 1 background
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

# Train the model
train(
    config, target_info, tomo_algorithm, voxel_size, loss_function,
    model_config, model_save_path=results_folder
)
```

### 3. Run Segmentation

```python
from octopi.workflows import segment

# Model paths from training
model_weights = f'{results_folder}/best_model.pth'
model_config = f'{results_folder}/model_config.yaml'

# Segmentation parameters
seg_info = ['predict', 'octopi', '1']

# Run segmentation
segment(
    config=config,
    tomo_algorithm=tomo_algorithm,
    voxel_size=voxel_size,
    model_weights=model_weights,
    model_config=model_config,
    seg_info=seg_info,
    use_tta=True
)
```

### 4. Extract Coordinates

```python
from octopi.workflows import localize

# Localization parameters
pick_user_id = 'octopi'
pick_session_id = '1'

# Run localization
localize(
    config=config,
    voxel_size=voxel_size,
    seg_info=seg_info,
    pick_user_id=pick_user_id,
    pick_session_id=pick_session_id
)
```

### 5. Evaluate Results

```python
from octopi.workflows import evaluate

# Evaluation against ground truth
evaluate(
    config=config,
    gt_user_id='data-portal',  # Ground truth source
    gt_session_id='1',
    pred_user_id=pick_user_id,
    pred_session_id=pick_session_id,
    distance_threshold=0.5,
    save_path=f'{results_folder}/evaluation'
)
```

<details markdown="1">
<summary><strong>💡 All in one script</strong></summary>

Copy and paste this complete script, then modify the configuration variables at the top:

```python
from octopi.entry_points.run_create_targets import create_sub_train_targets
from octopi.workflows import train, segment, localize, evaluate
from monai.losses import TverskyLoss

# =============================================================================
# CONFIGURATION - Modify these variables for your dataset
# =============================================================================

config = 'config.json'                    # Path to your Copick config
voxel_size = 10.012                       # Voxel size in Angstroms
tomo_algorithm = 'denoised'               # Tomogram algorithm identifier
results_folder = 'model_output'          # Where to save results

# Define your objects and annotation sources
pick_targets = [
    ('ribosome', 'data-portal', '1'),
    ('virus-like-particle', 'data-portal', '1'),
    ('apoferritin', 'data-portal', '1')
]

# Ground truth for evaluation
gt_user_id = 'data-portal'
gt_session_id = '1'

# =============================================================================
# WORKFLOW - No need to modify below this line
# =============================================================================

print("Step 1: Creating training targets...")
create_sub_train_targets(
    config, pick_targets, [], voxel_size, 0.7,
    tomo_algorithm, 'targets', 'octopi', '1'
)

print("Step 2: Training model...")
model_config = {
    'architecture': 'Unet',
    'num_classes': len(pick_targets) + 1,  # objects + background
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
    config, ['targets', 'octopi', '1'], tomo_algorithm, voxel_size,
    loss_function, model_config, model_save_path=results_folder
)

print("Step 3: Running segmentation...")
segment(
    config=config,
    tomo_algorithm=tomo_algorithm,
    voxel_size=voxel_size,
    model_weights=f'{results_folder}/best_model.pth',
    model_config=f'{results_folder}/model_config.yaml',
    seg_info=['predict', 'octopi', '1'],
    use_tta=True
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
</details>

## Key Parameters to Modify

- **`config`**: Path to your Copick configuration file
- **`voxel_size`**: Tomogram resolution (check your data specifications)
- **`tomo_algorithm`**: Algorithm used for tomogram reconstruction
- **`pick_targets`**: List of (object_name, user_id, session_id) for your annotations
- **`num_classes`**: Number of object types + 1 for background
- **`gt_user_id/gt_session_id`**: Ground truth annotation source for evaluation


## Next Steps

**For detailed explanations and advanced options:**

- **[Training Guide](training.md)** - Learn about loss functions, cross-validation, and model exploration
- **[Inference Guide](inference.md)** - Understand segmentation, localization, and evaluation in detail
- **[Adding New Models](adding-new-models.md)** - Integrate custom architectures