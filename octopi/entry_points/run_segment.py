from octopi.entry_points import common
from typing import List, Tuple
import rich_click as click

def inference(
    copick_config_path: str,
    model_weights: str, 
    model_config: str,
    seg_info: Tuple[str,str,str],
    voxel_size: float,
    tomo_algorithm: str,
    tomo_batch_size: int,
    run_ids: List[str],
    ):
    """
    Perform segmentation inference using a model on provided tomograms.

    Args:
        copick_config_path (str): Path to CoPick configuration file.
        run_ids (List[str]): List of tomogram run IDs for inference.
        model_weights (str): Path to the trained model weights file.
        channels (List[int]): List of channel sizes for each layer.
        strides (List[int]): List of strides for the layers.
        res_units (int): Number of residual units for the model.
        voxel_size (float): Voxel size for tomogram reconstruction.
        tomo_algorithm (str): Tomogram reconstruction algorithm to use.
        segmentation_name (str): Name for the segmentation output.
        segmentation_user_id (str): User ID associated with the segmentation.
        segmentation_session_id (str): Session ID for this segmentation run.
    """
    from octopi.pytorch import segmentation
    import torch
    
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    if ',' in model_weights:
        model_weights = model_weights.split(',')
    if ',' in model_config:
        model_config = model_config.split(',')
    if isinstance(model_weights, list) and isinstance(model_config, list):
        if len(model_weights) != len(model_config):
            raise ValueError("Number of model weights and model configs must match for ensemble prediction.")
        print("\nUsing Model Ensemble (Soup) Segmentation.")
        print('Model Weights:', model_weights)
        print('Model Configs:', model_config)
    else:
        print("Using Single Model Segmentation.")

    if gpu_count > 1:
        print("Using Multi-GPU Predictor.")
        predict = segmentation.MultiGPUPredictor(
            copick_config_path,
            model_config,
            model_weights
        )

        # Run Multi-GPU inference
        predict.multi_gpu_inference(
            runIDs=run_ids,
            tomo_algorithm=tomo_algorithm,
            voxel_spacing=voxel_size,
            segmentation_name=seg_info[0],
            segmentation_user_id=seg_info[1],
            segmentation_session_id=seg_info[2],
            save=True
        )

    else:
        print("Using Single-GPU Predictor.")
        predict = segmentation.Predictor(
            copick_config_path,
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

    print("Inference completed successfully.")


@click.command('segment')
# Inference Arguments
@common.inference_parameters()
# Model Arguments
@common.inference_model_parameters()
# Input Arguments
@common.config_parameters(single_config=True)
def cli(config, voxel_size,
        model_config, model_weights,
        tomo_alg, seg_info, tomo_batch_size, run_ids):
    """
    Segment volumes using trained neural network models.
    
    It supports both single model inference and model ensembles 
    (model soups) for improved accuracy. Multi-GPU inference is automatically enabled when 
    multiple GPUs are available.
    
    The segmentation masks are saved as zarr arrays in your copick project, organized by 
    segmentation name, user ID, and session ID for easy tracking and comparison.
    
    \b
    Examples:
      # Segment with a single model
      octopi segment -c config.json \\
        --model-config model.yaml --model-weights model.pth \\
        --seg-info predictions,octopi,1
    
    \b
      # Segment with model ensemble (comma-separated)
      octopi segment -c config.json \\
        --model-config model1.yaml,model2.yaml \\
        --model-weights model1.pth,model2.pth \\
        --seg-info ensemble,octopi,1
    
    \b
      # Segment specific runs only
      octopi segment -c config.json \\
        --model-config model.yaml --model-weights model.pth \\
        --run-ids TS_001,TS_002,TS_003 \\
        --tomo-batch-size 10
    """
    
    # Set default values if not provided
    seg_info = list(seg_info)  # Convert tuple to list
    if seg_info[1] is None:
        seg_info[1] = "octopi"
    if seg_info[2] is None:
        seg_info[2] = "1"

    # Call the inference function with parsed arguments
    inference(
        copick_config_path=config,
        model_weights=model_weights,
        model_config=model_config,
        seg_info=seg_info,
        voxel_size=voxel_size,
        tomo_algorithm=tomo_alg,
        tomo_batch_size=tomo_batch_size,
        run_ids=run_ids,
    )


if __name__ == "__main__":
    cli()