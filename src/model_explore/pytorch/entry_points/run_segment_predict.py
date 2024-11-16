from model_explore.pytorch import segmentation
from typing import List
import torch, argparse

def inference(
    copick_config_path: str,
    run_ids: List[str],
    model_weights: str,
    channels: List[int],
    strides: List[int],
    res_units: int,
    voxel_size: float,
    tomo_algorithm: str,
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str
    ):

    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:
        predict = segmentation.MultiGPUPredictor(
            copick_config_path,
            model_weights,
            my_channels = channels,
            my_strides = strides,
            my_num_res_units = res_units)
        
        # Option 2: Predict and Save all the RunIDs
        predict.batch_predict(
            runIDs = run_ids,
            tomo_algorithm = tomo_algorithm,
            voxel_spacing = voxel_size,
            segmentation_name = segmentation_name,
            segmentation_user_id = segmentation_user_id,
            segmentation_session_id = segmentation_session_id )
    else:
        predict = segmentation.Predictor(
            copick_config_path,
            model_weights,
            my_channels = channels,
            my_strides = strides,
            my_num_res_units = res_units)
        
        predict.multi_gpu_inference()

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run segmentation predictions with specified model and configuration.")
    
    # Add arguments
    parser.add_argument("--config_path", type=str, default="/mnt/simulations/ml_challenge/ml_config.json",
                        help="Path to the configuration file.")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to the model weights file.")
    parser.add_argument("--channels", nargs="+", type=int, default=[32, 64, 128, 128],
                        help="List of channel sizes for each layer.")
    parser.add_argument("--strides", nargs="+", type=int, default=[2, 2, 1],
                        help="List of stride values for each layer.")
    parser.add_argument("--res_units", type=int, default=2,
                        help="Number of residual units.")
    parser.add_argument("--tomo_algorithm", type=str, default="wbp",
                        help="Tomogram reconstruction algorithm.")
    parser.add_argument("--run_ids", nargs="*", default=None,
                        help="List of run IDs for prediction. If not provided, all run IDs will be processed.")

    # Parse arguments
    args = parser.parse_args()

    inference()

