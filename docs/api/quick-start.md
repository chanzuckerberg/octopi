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

# Initialize the model
model_builder = builder.get_model(model_config['architecture'])
model = model_builder.build_model(model_config)
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
# Define loss function and metrics
loss_fn = losses.WeightedFocalTverskyLoss()
metric_fn = ConfusionMatrixMetric(include_background=True)

# Create trainer
trainer = trainer.ModelTrainer(
    model, device, loss_fn, 
    metric_fn, optimizer
)

# Start training
model_save_path = 'results'
trainer.train(
    data_generator, model_save_path, max_epochs = 1000,
    crop_size=model_config['dim_in'], best_metric=best_metric, verbose=True
)
```

## Model Exploration with Optuna

For automatic model architecture search we can simply the entire process. By default, Octopi will sample various loss functions, parameters for the given architecture, 

```python
from octopi.pytorch.model_search_submitter import ModelSearchSubmit

optimizer = ModelSearchSubmit(
    copick_config=config,
    target_name=name, target_user_id=user_id, target_session_id = session_id
    tomo_algorithm=tomo_algorithm, voxel_size=voxel_size, Nclass=Nclass,
    num_epochs=1000, num_trials = 100,
    model_type = 'UNet'
)

search.run_model_search()
```

## Perform Segmentation Inference

With a trained model in hand, it's time to move forward with making predictions. In this step, you will execute model inference using a checkpoint from a saved epoch, allowing you to evaluate the model's performance on your test dataset.

```python 

from octopi.pytorch import segmentation

########### Input Parameters ###########

# Copick Query for Tomograms to Run Inference On
config = "10440_config.json"
tomo_algorithm = 'wbp-denoised-denoiset-ctfdeconv'
voxel_size = 10.012

# Path to Trained Model
model_weights = 'results/best_model.pth'
model_config = 'results/training_parameters.yaml'

# Adjust this parameter based on available GPU and RAM space
tomo_batch_size = 15

# RunIDs to Run Inference On
run_ids = None

# Output Save Information (Segmentation Name, UserID, SessionID)
seg_info = ['predict', 'DeepFindET', '1']

print("Using Single-GPU Predictor.")
predict = segmentation.Predictor(
    config,
    model_config,
    model_weights,
)

# Run batch prediction
predict.batch_predict(
    runIDs=run_ids,
    num_tomos_per_batch=tomo_batch_size,
    tomo_algorithm=tomo_algorithm,
    voxel_spacing=voxel_size,
    segmentation_name=seg_info[0],
    segmentation_user_id=seg_info[1],
    segmentation_session_id=seg_info[2]
)
```

## Converting Segmentation Masks to 3D Coordinates

Each segmentation mask highlights regions in the tomogram where the model believes a specific macromolecule is present. To transform these continuous volumetric predictions into biologically meaningful particle coordinates, we apply a post-processing pipeline that includes:

	•	Thresholding the segmentation mask to isolate high-confidence regions
	•	Size filtering based on expected macromolecule volume (e.g., diameter or voxel count)
	•	Centroid extraction from connected components that match the size profile

This process enables us to generate a precise list of coordinates that can be directly compared to ground truth annotations or used as input for downstream structural analysis.

By tuning the filtering parameters (e.g., minimum/maximum particle size), the pipeline can be adapted to different targets such as ribosomes, proteasomes, or other cellular components.

```python
from octopi.extract import localize
from tqdm import tqdm
import copick

########### Input Parameters ###########

# Copick Query for Tomograms to Run Inference On
config = "10440_config.json"

# Voxel Size of Segmentation Maps
voxel_size = 10.012

# Information for the Referenced Segmentation Map 
seg_info = ['predict', 'DeepFindET', '1']

# Information for the Saved Pick Session
pick_user_id = 'octopi'; pick_session_id = '1'

# Information for the Localization Method
method = 'watershed'; filter_size = 10

# Save of Segmentation Spheres that are Valid for Coordinates
radius_min_scale = 0.5; radius_max_scale = 1.5

# RunIDs to Run Localization On
runIDs = None

# List of Objects to Localize 
# (We can Either Specify Specific Objects to Localize or None to Find All Objects)
pick_objects = None

# Load the Copick Config
root = copick.from_file(config) 

# Get objects that can be Picked
objects = [(obj.name, obj.label, obj.radius) for obj in root.pickable_objects if obj.is_particle]

# Verify each object has the required attributes
for obj in objects:
    if len(obj) < 3 or not isinstance(obj[2], (float, int)):
        raise ValueError(f"Invalid object format: {obj}. Expected a tuple with (name, label, radius).")

# Filter elements
if pick_objects is not None:
    objects = [obj for obj in objects if obj[0] in pick_objects]

print(f'Running Localization on the Following Objects: ')
print(', '.join([f'{obj[0]} (Label: {obj[1]})' for obj in objects]) + '\n')

# Either Specify Input RunIDs or Run on All RunIDs
if runIDs:  print('Running Localization on the Following RunIDs: ' + ', '.join(runIDs) + '\n')
run_ids = runIDs if runIDs else [run.name for run in root.runs]
n_run_ids = len(run_ids)

# Main Loop
for run_id in tqdm(run_ids):

    # Run Localization on Given RunID
    run = root.get_run(run_id)
    localize.processs_localization(
        run,
        objects,
        seg_info,
        method,
        voxel_size,
        filter_size,
        radius_min_scale, radius_max_scale,
        pick_session_id, pick_user_id,
    )

```