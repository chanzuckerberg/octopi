from model_explore.pytorch import segmentation, utils
from typing import List
import torch, argparse

def inference(
    copick_config_path: str,
    model_weights: str,
    segmentation_name: str = 'segment-predict',
    segmentation_user_id: str = 'monai',
    segmentation_session_id: str = '1',
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    channels: List[int] = [32,64,128,128],
    strides: List[int] = [2,2,1], 
    res_units: int = 2,
    run_ids: List[str] = None,
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

    if gpu_count > 1:
        print("Using Multi-GPU Predictor.")
        predict = segmentation.MultiGPUPredictor(
            copick_config_path,
            model_weights,
            my_channels=channels,
            my_strides=strides,
            my_num_res_units=res_units
        )

        # Run batch prediction
        predict.batch_predict(
            runIDs=run_ids,
            tomo_algorithm=tomo_algorithm,
            voxel_spacing=voxel_size,
            segmentation_name=segmentation_name,
            segmentation_user_id=segmentation_user_id,
            segmentation_session_id=segmentation_session_id
        )
    else:
        print("Using Single-GPU Predictor.")
        predict = segmentation.Predictor(
            copick_config_path,
            model_weights,
            my_channels=channels,
            my_strides=strides,
            my_num_res_units=res_units
        )

        # Run single GPU inference
        predict.multi_gpu_inference()

    print("Inference completed successfully.")

# Entry point with argparse
def cli():
    """
    CLI entry point for running inference.
    """
    parser = argparse.ArgumentParser(description="Run segmentation predictions with a specified model and configuration on CryoET Tomograms.")

    # Add arguments
    parser.add_argument("--config-path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model-weights", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--channels", type=utils.parse_int_list, required=False, default="32,64,128,128", help="List of channel sizes for each layer, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, required=False, default="2,2,1", help="List of stride values for each layer, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res-units", type=int, default=2, required=False, help="Number of residual units. Default is 2.")
    parser.add_argument("--voxel-size", type=float, default=10.0, required=False, help="Voxel size for tomogram reconstruction. Default is 10.0.")
    parser.add_argument("--tomo-algorithm", type=str, default="wbp", required=False, help="Tomogram reconstruction algorithm. Default is 'wbp'.")
    parser.add_argument("--segmentation-name", type=str, default="segment-predict", required=False, help="Name for the segmentation output. Default is 'segment-predict'.")
    parser.add_argument("--segmentation-user-id", type=str, default="monai", required=False, help="User ID associated with the segmentation. Default is 'monai'.")
    parser.add_argument("--segmentation-session-id", type=str, default="1", required=False, help="Session ID for the segmentation. Default is '1'.")
    parser.add_argument("--run-ids", type=utils.parse_list, default=None, required=False, help="List of run IDs for prediction, e.g., run1,run2 or [run1,run2]. If not provided, all available runs will be processed.")

    # Parse arguments
    args = parser.parse_args()

    # Call the inference function with parsed arguments
    inference(
        copick_config_path=args.config_path,
        model_weights=args.model_weights,
        channels=args.channels,
        strides=args.strides,
        res_units=args.res_units,
        voxel_size=args.voxel_size,
        tomo_algorithm=args.tomo_algorithm,
        segmentation_name=args.segmentation_name,
        segmentation_user_id=args.segmentation_user_id,
        segmentation_session_id=args.segmentation_session_id,
        run_ids=args.run_ids,
    )

if __name__ == "__main__":
    cli()
