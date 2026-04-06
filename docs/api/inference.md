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


=== "Single Tomogram"

    Load a model and apply it directly to a numpy array — useful for custom pipelines or interactive notebooks.

    ```python
    import numpy as np
    from octopi.pytorch.segmentation import Predictor

    model_weights = 'model_output/best_model.pth'
    model_config = 'model_output/model_config.yaml'

    # Initialize the predictor
    predictor = Predictor(
        config='config.json',
        model_config=model_config,
        model_weights=model_weights,
        ntta=4
    )

    # Load your tomogram as a numpy array [Z, Y, X]
    tomogram = np.load('my_tomogram.npy')

    # Run inference — returns a segmentation mask [Z, Y, X]
    segmentation = predictor.predict(tomogram)
    ```

    The returned array has the same spatial dimensions as the input, with integer values corresponding to each class label defined in your CoPick config.

=== "Full CoPick Dataset"

    Process all runs in a CoPick project and save segmentations back to the project structure.

    ```python
    from octopi.workflows import segment

    config = 'eval_config.json'
    model_weights = 'model_output/best_model.pth'
    model_config = 'model_output/model_config.yaml'
    seg_info = ['predict', 'octopi', '1']  # (name, user_id, session_id)

    segment(
        config=config,
        tomo_algorithm='denoised',
        voxel_size=10.0,
        model_weights=model_weights,
        model_config=model_config,
        seg_info=seg_info,
        ntta=4
    )
    ```

    <details markdown="1">
    <summary><strong>💡 segment() reference</strong></summary>

    `segment(config, tomo_algorithm, voxel_size, model_weights, model_config, seg_info=['predict', 'octopi', '1'], run_ids=None, batch_size=1, ntta=4)`

    **Parameters:**

    - `config` (str): Path to CoPick configuration file
    - `tomo_algorithm` (str): Tomogram algorithm identifier
    - `voxel_size` (float): Voxel spacing in Angstroms
    - `model_weights` (str or list): Path(s) to trained model weights (.pth file(s))
    - `model_config` (str or list): Path(s) to model configuration (.yaml file(s))
    - `seg_info` (list): Output segmentation identifier `(name, user_id, session_id)`
    - `run_ids` (list): Specific run IDs to process (default: None — all runs)
    - `batch_size` (int): Tomograms processed concurrently per GPU (default: 1)
    - `ntta` (int): Number of test-time augmentation rotations (default: 4; set to 1 to disable)

    </details>

## Localization

Extract particle coordinates from segmentation masks using watershed algorithm. The localization process converts probability maps into discrete particle coordinates by identifying local maxima and applying size constraints.

!!! info "Localization Parameters"
    - **seg_info**: Tuple specifying which segmentation to process `(name, user_id, session_id)`
    - **pick_user_id/pick_session_id**: Identifiers for saving the resulting coordinates
    - **voxel_size**: Voxel spacing for coordinate scaling
    - **method**: Particle detection algorithm (watershed by default)

The algorithm automatically applies size filtering based on object radii defined in your Copick configuration, using scale factors of 0.4-1.0 times the expected radius.


=== "Single Tomogram"

    Apply localization directly to a segmentation numpy array — no CoPick project required.

    ```python
    import numpy as np
    from octopi.extract.localize import extract_coordinates

    # Integer-labeled segmentation array [Z, Y, X] — e.g. output of predictor.predict()
    segmentation = np.load('segmentation.npy')

    voxel_size = 10.0  # Angstroms per voxel

    # Define objects: (name, label, radius_in_angstroms)
    objects = [
        ('ribosome',            1, 150.0),
        ('virus-like-particle', 2, 500.0),
        ('apoferritin',         3,  60.0),
    ]

    # Call extract_coordinates per particle class
    # min_radius / max_radius are in voxels (radius_angstrom / voxel_size)
    for name, label, radius in objects:
        points = extract_coordinates(
            seg=segmentation,
            min_radius=0.5 * radius / voxel_size,
            max_radius=1.0 * radius / voxel_size,
            label=label,
            method='watershed',
            filter_size=10,
        )
        # points — np.ndarray of shape (N, 3) in voxel coordinates [Z, Y, X]
        print(f'{name}: {len(points)} particles')
    ```

=== "Full CoPick Dataset"

    Localize all runs in a CoPick project in parallel across CPU cores.

    ```python
    from octopi.workflows import localize

    localize(
        config='config.json',
        voxel_size=10.0,
        seg_info=['predict', 'octopi', '1'],
        pick_user_id='octopi',
        pick_session_id='1'
    )
    ```

    <details markdown="1">
    <summary><strong>💡 localize() reference</strong></summary>

    `localize(config, voxel_size, seg_info, pick_user_id, pick_session_id, n_procs=16, method='watershed', filter_size=10, radius_min_scale=0.4, radius_max_scale=1.0, run_ids=None, pick_objects=None)`

    Extracts particle coordinates from segmentation masks using watershed or center-of-mass detection, parallelized across runs. Reads segmentations from and writes picks back to the CoPick project.

    **Parameters:**

    - `config` (str): Path to CoPick configuration file
    - `voxel_size` (float): Voxel spacing in Angstroms
    - `seg_info` (list): Segmentation to process `(name, user_id, session_id)`
    - `pick_user_id` (str): User ID for output coordinates
    - `pick_session_id` (str): Session ID for output coordinates
    - `n_procs` (int): Number of parallel CPU processes (default: 16)
    - `method` (str): Detection algorithm — `'watershed'` or `'com'` (default: `'watershed'`)
    - `filter_size` (int): Gaussian filter size for watershed smoothing (default: 10)
    - `radius_min_scale` (float): Minimum particle size as fraction of config radius (default: 0.4)
    - `radius_max_scale` (float): Maximum particle size as fraction of config radius (default: 1.0)
    - `run_ids` (list): Specific run IDs to process (default: None — all runs)
    - `pick_objects` (list): Specific object names to localize (default: None — all objects)

    !!! tip "CoPick-free usage"
        Use `extract_coordinates(seg, min_radius, max_radius, label, method, filter_size)` from `octopi.extract.localize` to localize a single segmentation array directly — returns an `np.ndarray` of shape `(N, 3)` in voxel coordinates `[Z, Y, X]`. See the **Single Tomogram** tab above.

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

??? example "Running Evaluation"

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