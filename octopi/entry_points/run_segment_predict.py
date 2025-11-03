import torch, argparse, json, pprint, yaml, os
from octopi.pytorch import segmentation
from octopi.entry_points import common
from typing import List, Tuple
from octopi.utils import io

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

def inference_parser(parser_description, add_slurm: bool = False):
    """
    Parse the arguments for the inference
    """
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_argument_group("Input Arguments")
    common.add_config(input_group, single_config=True)

    model_group = parser.add_argument_group("Model Arguments")
    common.inference_model_parameters(model_group)

    inference_group = parser.add_argument_group("Inference Arguments")
    common.add_inference_parameters(inference_group)

    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'segment_predict', gpus = 2)

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    """
    CLI entry point for running inference.
    """
    
    # Parse the arguments
    parser_description = "Run segmentation predictions with a specified model and configuration on CryoET Tomograms."
    args = inference_parser(parser_description)

    # Set default values if not provided
    args.seg_info = list(args.seg_info)  # Convert tuple to list
    if args.seg_info[1] is None:
        args.seg_info[1] = "octopi"

    if args.seg_info[2] is None:
        args.seg_info[2] = "1"

    # Call the inference function with parsed arguments
    inference(
        copick_config_path=args.config,
        model_weights=args.model_weights,
        model_config=args.model_config,
        seg_info=args.seg_info,
        voxel_size=args.voxel_size,
        tomo_algorithm=args.tomo_alg,
        tomo_batch_size=args.tomo_batch_size,
        run_ids=args.run_ids,
    )