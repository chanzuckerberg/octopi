from model_explore.pytorch import segmentation, utils
import torch, argparse, json
from typing import List

def inference(
    copick_config_path: str,
    model_weights: str, 
    model_type: str = 'Unet',
    segmentation_name: str = 'segment-predict',
    segmentation_user_id: str = 'monai',
    segmentation_session_id: str = '1',
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    channels: List[int] = [32,64,128,128],
    strides: List[int] = [2,2,1],
    res_units: int = 2,
    dim_in: int = 96,
    nclass: int = 3,    
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
            model_type=model_type,
            my_channels=channels,
            my_strides=strides,
            my_num_res_units=res_units,
            my_nclass=nclass,
            dim_in=dim_in
        )

        # Run Multi-GPU inference
        predict.multi_gpu_inference(
            runIDs=run_ids,
            tomo_algorithm=tomo_algorithm,
            voxel_spacing=voxel_size,
            segmentation_name=segmentation_name,
            segmentation_user_id=segmentation_user_id,
            segmentation_session_id=segmentation_session_id,
            save=True
        )

    else:
        print("Using Single-GPU Predictor.")
        predict = segmentation.Predictor(
            copick_config_path,
            model_weights,
            model_type=model_type,
            my_channels=channels,
            my_strides=strides,
            my_num_res_units=res_units,
            my_nclass=nclass,
            dim_in=dim_in
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

    print("Inference completed successfully.")

# Entry point with argparse
def cli():
    """
    CLI entry point for running inference.
    """
    parser = argparse.ArgumentParser(description="Run segmentation predictions with a specified model and configuration on CryoET Tomograms.")

    # Add arguments
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model-weights", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--model-type", type=str, default="Unet", required=False, help="Type of model to use. Available options: ['UNet', 'AttentionUnet']. Default is 'UNet'.")
    parser.add_argument("--Nclass", type=int, default=3, required=False, help="Number of classes. Default is 3.")
    parser.add_argument("--channels", type=utils.parse_int_list, required=False, default="32,64,128,128", help="List of channel sizes for each layer, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, required=False, default="2,2,1", help="List of stride values for each layer, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res-units", type=int, default=2, required=False, help="Number of residual units. Default is 2.")
    parser.add_argument("--dim-in", type=int, default=96, required=False, help="Dimension of the input tomograms. Default is 96.")
    parser.add_argument("--voxel-size", type=float, default=10.0, required=False, help="Voxel size for tomogram reconstruction. Default is 10.0.")
    parser.add_argument("--tomogram-algorithm", type=str, default="wbp", required=False, help="Tomogram reconstruction algorithm. Default is 'wbp'.")
    parser.add_argument("--segmentation-name", type=str, default="segment-predict", required=False, help="Name for the segmentation output. Default is 'segment-predict'.")
    parser.add_argument("--segmentation-user-id", type=str, default="monai", required=False, help="User ID associated with the segmentation. Default is 'monai'.")
    parser.add_argument("--segmentation-session-id", type=str, default="1", required=False, help="Session ID for the segmentation. Default is '1'.")
    parser.add_argument("--run-ids", type=utils.parse_list, default=None, required=False, help="List of run IDs for prediction, e.g., run1,run2 or [run1,run2]. If not provided, all available runs will be processed.")
    
    # Parse arguments
    args = parser.parse_args()

    # Save JSON with Parameters
    output_json = f'segment-predict_{args.segmentation_user_id}_{args.segmentation_session_id}_{args.segmentation_name}.json'
    save_parameters_json(args, output_json)       

    # Call the inference function with parsed arguments
    inference(
        copick_config_path=args.config,
        model_weights=args.model_weights,
        model_type=args.model_type,
        channels=args.channels,
        strides=args.strides,
        res_units=args.res_units,
        nclass=args.Nclass,
        dim_in=args.dim_in,
        voxel_size=args.voxel_size,
        tomo_algorithm=args.tomogram_algorithm,
        segmentation_name=args.segmentation_name,
        segmentation_user_id=args.segmentation_user_id,
        segmentation_session_id=args.segmentation_session_id,
        run_ids=args.run_ids,
    )

def save_parameters_json(args: argparse.Namespace, 
                         output_path: str):    

    # Create parameters dictionary
    params = {
        "inputs": {
            "config": args.config,
            "model_weights": args.model_weights,
            "tomo_algorithm": args.tomogram_algorithm,
            "voxel_size": args.voxel_size
        },
        "outputs": {
            "segmentation_name": args.segmentation_name,
            "segmentation_user_id": args.segmentation_user_id,
            "segmentation_session_id": args.segmentation_session_id
        },
        "model": {
            "model_type": args.model_type,
            "channels": args.channels,
            "strides": args.strides,
            "res_units": args.res_units,
            "nclass": args.Nclass,
            "dim_in": args.dim_in,
            "run_ids": args.run_ids
        }
    }                         

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)    

if __name__ == "__main__":
    cli()
