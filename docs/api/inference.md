# Inference

This page covers running inference with trained octopi models, including segmentation, localization, and evaluation.

## Segmentation

Run trained models on tomograms to generate segmentation masks. The segmentation process takes your trained model weights and configuration to produce probability maps for each object class defined in your training targets.

!!! info "Segmentation Parameters"
    - **model_weights**: Path to your trained model weights (`.pth` file from training)
    - **model_config**: Path to model configuration (`.yaml` file from training)
    - **seg_info**: Tuple defining where to save segmentation results (`name`, `user_id`, `session_id`)
    - **use_tta**: Whether to use test-time augmentation for improved robustness
    - **run_ids**: Optional list of specific tomograms to process (None for all available)

The algorithm automatically applies size filtering based on object radii defined in your Copick configuration, using scale factors of 0.4-1.0 times the expected radius.

!!! tip 
    We can enable **model soup ensembling** by providing a list of model weights (or optionally a single or list of model configs). 
    
    This technique averages predictions from multiple models, which is particularly useful for combining cross-validation fold models or experimenting with different architectures to improve robustness and reduce prediction variance.


### Running Segmentation

```python
from octopi.workflows import segment

# Configuration
config = 'eval_config.json'
model_weights = 'model_output/best_model.pth'
model_config = 'model_output/model_config.yaml'

# Segmentation parameters
seg_info = ['predict', 'octopi', '1']  # (name, user_id, session_id)
tomo_algorithm = 'denoised'
voxel_size = 10.0

# Run segmentation
segment(
    config=config,
    tomo_algorithm=tomo_algorithm,
    voxel_size=voxel_size,
    model_weights=model_weights,
    model_config=model_config,
    seg_info=seg_info,
    use_tta=True  # Test-time augmentation
)
```

<details markdown="1">
<summary><strong>💡 Segmentation Function Reference</strong></summary>

`segment(config, tomo_algorithm, voxel_size, model_weights, model_config, seg_info=['predict', 'octopi', '1'], use_tta=False)`

The segmentation process applies your trained model(s) to tomograms in batches for memory efficiency. It supports both single model inference and Model Soup ensembling, with optional test-time augmentation. Predictions are averaged across models and augmentations, then converted to discrete segmentation masks.

**Parameters:**

- `config` (str): Path to Copick configuration file
- `tomo_algorithm` (str): Tomogram algorithm identifier  
- `voxel_size` (float): Voxel spacing in Angstroms
- `model_weights` (str or list): Path(s) to trained model weights (.pth file(s))
- `model_config` (str or list): Path(s) to model configuration (.yaml file(s))
- `seg_info` (tuple): Segmentation specification `(name, user_id, session_id)` (default: ['predict', 'octopi', '1'])
- `use_tta` (bool): Enable test-time augmentation (default: False)

**Outputs:**

Segmentation masks saved to Copick structure with probability values for each object class.
</details>

## Localization

Extract particle coordinates from segmentation masks using watershed algorithm. The localization process converts probability maps into discrete particle coordinates by identifying local maxima and applying size constraints.

!!! info "Localization Parameters"
    - **seg_info**: Tuple specifying which segmentation to process `(name, user_id, session_id)`
    - **pick_user_id/pick_session_id**: Identifiers for saving the resulting coordinates
    - **voxel_size**: Voxel spacing for coordinate scaling
    - **method**: Particle detection algorithm (watershed by default)

The algorithm automatically applies size filtering based on object radii defined in your Copick configuration, using scale factors of 0.4-1.0 times the expected radius.


### Running Localization

```python
from octopi.workflows import localize

# Localization parameters
seg_info = ['predict', 'octopi', '1']  # Segmentation to process
pick_user_id = 'octopi'
pick_session_id = '1'
voxel_size = 10.0

# Run localization
localize(
    config=config,
    voxel_size=voxel_size,
    seg_info=seg_info,
    pick_user_id=pick_user_id,
    pick_session_id=pick_session_id
)
```

<details markdown="1">
<summary><strong>💡 Localization Function Reference</strong></summary>

`localize(config, voxel_size, seg_info, pick_user_id, pick_session_id, method = 'watershed', filter_size = 10, radius_min_scale = 0.4, radius_max_scale = 1.0)`

Extracts particle coordinates from segmentation masks using watershed algorithm. This method uses $N$ parallel processes for efficient processing. We can either applies watershed algorithm with Gaussian filtering (filter_size=10) or measure each unique objects center of mass. We will filter the particles by size defined by the input parameters `radius_min_scale` and `radius_max_scale`. 

**Parameters:**

- `config` (str): Path to Copick configuration file
- `voxel_size` (float): Voxel spacing in Angstroms
- `seg_info` (tuple): Segmentation to process `(name, user_id, session_id)`
- `pick_user_id` (str): User ID for output coordinates
- `pick_session_id` (str): Session ID for output coordinates

**Algorithm Details:**

- **Method**: Watershed segmentation
- **Filter size**: 10 (Gaussian filter for smoothing)
- **Radius constraints**: 0.4-1.0 × object radius from config
- **Parallel processing**: 32 processes (adjustable based on system)

**Outputs:**

Particle coordinates saved to Copick structure in standard format.

</details>

## Evaluation

Compare predicted coordinates against ground truth annotations to assess model performance. The evaluation process calculates standard metrics including **precision**, **recall**, and $F_{\beta}$-score using distance-based matching. The evaluation metrics are defined as:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

$$F_{\beta} = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$

!!! info "Evaluation Parameters"
    - **gt_user_id/gt_session_id**: Ground truth annotation identifiers
    - **pred_user_id/pred_session_id**: Predicted coordinate identifiers
    - **distance_threshold**: Normalized distance threshold for matching (default: 0.5)
    - **run_ids**: Optional list of specific tomograms to evaluate (None for all available)

### Running Evaluation

```python
from octopi.workflows import evaluate

# Evaluation parameters
gt_user_id = 'curation'
gt_session_id = '0'
pred_user_id = 'octopi'
pred_session_id = '1'
distance_threshold = 0.5

# Run evaluation
evaluate(
    config=config,
    gt_user_id=gt_user_id,
    gt_session_id=gt_session_id,
    pred_user_id=pred_user_id,
    pred_session_id=pred_session_id,
    distance_threshold=distance_threshold,
    save_path='evaluation_results'
)
```