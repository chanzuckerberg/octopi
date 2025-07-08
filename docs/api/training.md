# Training

This page covers data preparation and training deep learning models for 3D particle picking using octopi.

## Target Creation

Before training, you need to create training targets from existing particle annotations. We can explicitly define all the objects we'd like to query with a list of tuples `(object_name, user_id, session_id)` defining source annotations. This query can either be for point coordinates for protein coordinates, or continuous segmentations such as for membranes or organelles generated from software such as [membrain-seg](https://github.com/teamtomo/membrain-seg) or saber.

The target creation process uses the following key parameters:

* **pick_targets**: List of tuples (object_name, user_id, session_id) defining source point annotations
* **seg_targets**: List of segmentation targets (same format as pick_targets) for continuous structures
* **radius_scale**: Scale factor for creating spherical targets relative to object radius defined in config
* **run_ids**: Optional subset of tomograms (None for all available)

<details>
<summary><strong>💡 Seeing Available Queries Per Copick Project </strong></summary>
Remember, we can see all the available runs and queries in every copick project with the following CLI command:

For local projects:
```bash
copick browse -c config.json
```

For training data from the data-portal:
```bash
copick browse -ds datasetID
```

</details>

### Creating Training Targets

```python
from octopi.entry_points.run_create_targets import create_sub_train_targets

# Configuration
config = '../config.json'
target_name = 'targets'
target_user_id = 'octopi'
target_session_id = '1'

# Tomogram parameters
voxel_size = 10.012
tomogram_algorithm = 'wbp-denoised-denoiset-ctfdeconv'
radius_scale = 0.7  # Fraction of particle radius for sphere targets

# Define source annotations
pick_targets = [
    ('ribosome', 'data-portal', '1'),
    ('virus-like-particle', 'data-portal', '1'),
    ('apoferritin', 'data-portal', '1')
]

seg_targets = ['membrane', 'membrain-seg', '2']  # Optional segmentation targets

# Create targets
create_sub_train_targets(
    config, pick_targets, seg_targets, voxel_size, radius_scale, 
    tomogram_algorithm, target_name, target_user_id, target_session_id
)
```

## Model Training

Once training targets are created, you can train deep learning models for particle segmentation. The training process requires defining data splits, model architecture, loss functions, and training parameters.
Key training parameters include:

* **target_info**: Tuple specifying which training targets to use (name, user_id, session_id)
* **trainRunIDs/validateRunIDs**: Optional lists defining which tomograms to use for training vs validation. If None, octopi automatically uses all runs that have the desired segmentation target.
* **model_config**: Dictionary specifying the neural network architecture and parameters. We have a few default models to chose from, including the opportunity to [import new model architectures](adding-new-models.md).
* **loss_function**: Loss function appropriate for segmentation tasks (handles class imbalance)
* **use_ema**: Whether to use exponential moving average for more stable training

### Basic Training Setup

```python
from octopi.losses import FocalTverskyLoss
from monai.losses import TverskyLoss
from octopi.workflows import train

# Training configuration
config = 'train_config.json'
target_info = ['targets', 'octopi', '1']
results_folder = 'model_output'

# Data splits
trainRunIDs = ['run_001', 'run_002', 'run_003']
valRunIDs = ['run_004']

# Model architecture
model_config = {
    'architecture': 'Unet',
    'num_classes': 6,   # number of objects + 1 for background
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
    config, target_info, 'denoised', 10.012, loss_function,
    model_config, trainRunIDs=trainRunIDs, validateRunIDs=valRunIDs
)
```

<details markdown="1">
<summary><strong>💡 Training Function Reference  </strong></summary>

`train(config, target_info, tomo_algorithm, voxel_size, loss_function,`
`model_config, model_weights = None, trainRunIDs = None, validateRunIDs = None,`
`model_save_path = 'results', best_metric = 'fBeta2', num_epochs = 1000, use_ema = True)`

Trains a segmentation model for particle detection.

**Parameters:**

- `config` (str): Path to Copick configuration file
- `target_info` (tuple): Target specification `(name, user_id, session_id)`
- `tomo_algorithm` (str): Tomogram algorithm identifier
- `voxel_size` (float): Voxel spacing in Angstroms
- `loss_function`: PyTorch loss function
- `model_config` (dict): Model architecture configuration
- `model_weights` (str): Path to pretrained weights (default: None)
- `trainRunIDs` (list): List of training tomogram IDs (default: None - uses all available)
- `validateRunIDs` (list): List of validation tomogram IDs (default: None - uses all available)
- `model_save_path` (str): Directory to save trained model (default: 'results')
- `best_metric` (str): Metric for model selection (default: 'fBeta2')
- `num_epochs` (int): Maximum training epochs (default: 1000)
- `use_ema` (bool): Use exponential moving average (default: True)

**Returns:**

Training results dictionary with loss and metric histories.

**Outputs:**

- `{model_save_path}/best_model.pth`: Best model weights
- `{model_save_path}/model_config.yaml`: Model configuration
- `{model_save_path}/results.json`: Training metrics and results

</details>

### Cross Validation

Cross-validation is essential for robust model evaluation, especially when working with limited tomography data. It helps assess how well your model generalizes to unseen tomograms and prevents overfitting to specific data characteristics. This approach systematically trains multiple models, each time holding out a different tomogram for validation.

```python
# Cross-validation training
def train_cross_validation(split_id):
    runIDs = get_all_run_ids()
    valRunID = [runIDs[split_id]]
    trainRunIDs = runIDs[:split_id] + runIDs[split_id+1:]
    
    results_folder = f'split_{split_id}'
    
    train(
        ...
        trainRunIDs=trainRunIDs,
        validateRunIDs=valRunID,
        ...
    )
```

### Loss Functions

Choosing the right loss function is crucial for effective particle segmentation, especially given the class imbalance typical in cryo-ET data where background voxels vastly outnumber particle voxels.

**[MONAI Loss Functions](https://docs.monai.io/en/stable/losses.html):**
```python
from monai.losses import FocalLoss, TverskyLoss, DiceLoss, GeneralizedDiceLoss
```

- **TverskyLoss**: Generalizes Dice loss with separate weights for false positives (`alpha`) and false negatives (`beta`). Excellent for imbalanced data - use higher `beta` (e.g., 0.7) to penalize missed particles more than false detections.
- **DiceLoss**: Standard choice for segmentation, works well when classes are relatively balanced.

**Octopi Custom Loss Functions:**
```python
from octopi.losses import FocalTverskyLoss, WeightedFocalTverskyLoss
```

- **FocalTverskyLoss**: Combines Tversky's class balancing with Focal's hard example mining. The `gamma` parameter adds focusing power to the Tversky formulation.
- **WeightedFocalTverskyLoss**: Weighted combination of separate Focal and Tversky losses, allowing fine-tuned control over both class imbalance and hard example emphasis through `weight_tversky` and `weight_focal` parameters.

## Model Exploration

In cases where we'd like to automatically explore the model architecture landscape to determine which model configuration would be optimal for our given experiment, we can use the `ModelSearchSubmit` class. Here, the Bayesian optimizer will explore various loss functions and their associated hyperparameters, as well as architecture parameters which are defined in the model class. This automated approach significantly reduces the number of input parameters you need to provide, as the system intelligently searches through the hyperparameter space.

The automated search process uses Optuna's Bayesian optimization to efficiently explore combinations of:

- **Loss function types** and their hyperparameters (alpha, beta, gamma values)
- **Model architecture parameters** (channel sizes, dropout rates, number of residual units)

This is particularly valuable when working with new datasets or particle types where optimal configurations are unknown.

```python
from octopi.pytorch.model_search_submitter import ModelSearchSubmit

config = 'config.json'
target_name = 'targets'
target_user_id = 'octopi'
target_session_id = '1'
tomo_algorithm = 'denoised'
voxel_size = 10
Nclass = 6 # number of objects + 1 for background

optimizer = ModelSearchSubmit(
    config, target_name, target_user_id, target_session_id,
    tomo_algorithm, voxel_size, Nclass, 'UNet' )

optimizer.run_model_search()
```

<details markdown="1">
<summary><strong>💡 ModelSearchSubmit Class Reference</strong></summary>

`ModelSearchSubmit(copick_config, target_name, target_user_id, target_session_id, tomo_algorithm, voxel_size, Nclass, model_type, best_metric='avg_f1', num_epochs=1000, num_trials=100, data_split=0.8, random_seed=42, val_interval=10, tomo_batch_size=15, trainRunIDs=None, validateRunIDs=None, mlflow_experiment_name='explore')`

Initialize the ModelSearch class for architecture search with Optuna.

**Parameters:**

- `copick_config` (str or dict): Path to the CoPick configuration file or a dictionary for multi-config training
- `target_name` (str): Name of the target for segmentation
- `target_user_id` (str): User ID for target tracking
- `target_session_id` (str): Session ID for target tracking
- `tomo_algorithm` (str): Tomogram algorithm to use
- `voxel_size` (float): Voxel size for tomograms
- `Nclass` (int): Number of prediction classes
- `model_type` (str): Type of model to use (e.g., 'UNet')
- `best_metric` (str): Metric to optimize (default: 'avg_f1')
- `num_epochs` (int): Number of epochs per trial (default: 1000)
- `num_trials` (int): Number of trials for hyperparameter optimization (default: 100)
- `data_split` (float): Data split ratio for train/validation (default: 0.8)
- `random_seed` (int): Seed for reproducibility (default: 42)
- `val_interval` (int): Validation interval during training (default: 10)
- `tomo_batch_size` (int): Batch size for tomogram loading (default: 15)
- `trainRunIDs` (List[str]): List of training run IDs (default: None - uses all available)
- `validateRunIDs` (List[str]): List of validation run IDs (default: None - uses all available)
- `mlflow_experiment_name` (str): MLflow experiment name for tracking (default: 'explore')

**Methods:**

- `run_model_search()`: Executes the Bayesian optimization search across the defined parameter space

**Outputs:**

- MLflow experiment logs with trial results and metrics
- Best model configuration and weights
- Hyperparameter optimization history and visualizations

</details>

## Next Steps

Once you've successfully trained your model, you're ready to move to the inference stage. The trained model outputs (`.pth` weights and `.yaml` configuration) from this training process will be used directly in the inference pipeline.

**Ready to apply your trained model?** Head to the [**Inference Guide**](inference.md) to learn how to:

- Run segmentation on new tomograms using your trained weights
- Perform particle localization from segmentation masks
- Evaluate your model's performance against ground truth data

**Want to experiment with custom architectures?** Check out the [**Adding New Models**](adding-new-models.md) template to:

- Implement custom neural network architectures
- Integrate new model types into the octopi framework
- Extend the automated model search capabilities

**Pro tip:** If you're unsure about optimal hyperparameters, consider running the automated model exploration first before manual training - it can save significant time and often discovers configurations you might not have considered!