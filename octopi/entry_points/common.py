from octopi.utils import parsers
import argparse

def add_model_parameters(parser, octopi = False):
    """
    Add common model parameters to the parser.
    """
    
    # Add U-Net model parameters
    parser.add_argument("--Nclass", type=int, required=False, default=3, help="Number of prediction classes in the model")
    parser.add_argument("--channels", type=parsers.parse_int_list, required=False, default='32,64,96,96', help="List of channel sizes")
    parser.add_argument("--strides", type=parsers.parse_int_list, required=False, default='2,2,1', help="List of stride sizes")
    parser.add_argument("--res-units", type=int, required=False, default=1, help="Number of residual units in the UNet")
    parser.add_argument("--dim-in", type=int, required=False, default=96, help="Input dimension for the UNet model")

def inference_model_parameters(parser):
    """
    Add model parameters for inference.
    """
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--model-weights", type=str, required=True, help="Path to the model weights file")

def add_train_parameters(parser, octopi = False):
    """
    Add training parameters to the parser.
    """
    parser.add_argument("--num-epochs", type=int, required=False, default=1000, help="Number of training epochs")
    parser.add_argument("--val-interval", type=int, required=False, default=10, help="Interval for validation metric calculations")
    parser.add_argument("--tomo-batch-size", type=int, required=False, default=15, help="Number of tomograms to load per epoch for training")
    parser.add_argument("--best-metric", type=str, default='avg_f1', required=False, help="Metric to Monitor for Determining Best Model. To track fBetaN, use fBetaN with N as the beta-value.")
    
    if not octopi:
        parser.add_argument("--num-tomo-crops", type=int, required=False, default=16, help="Number of tomogram crops to use per patch")    
        parser.add_argument("--lr", type=float, required=False, default=1e-3, help="Learning rate for the optimizer")
        parser.add_argument("--tversky-alpha", type=float, required=False, default=0.3, help="Alpha parameter for the Tversky loss")
        parser.add_argument("--model-save-path", required=False, default='results', help="Path to model save directory")
    else:
        parser.add_argument("--num-trials", type=int, default=10, required=False, help="Number of trials for architecture search (default: 10).")

            
def add_config(parser, single_config):
    if single_config:
        parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    else:
        parser.add_argument("--config", type=str, required=True, action='append',
                            help="Specify a single configuration path (/path/to/config.json) "
                                 "or multiple entries in the format session_name,/path/to/config.json. "
                                 "Use multiple --config entries for multiple sessions.")
    parser.add_argument("--voxel-size", type=float, required=False, default=10, help="Voxel size of tomograms used")

def add_inference_parameters(parser):

    parser.add_argument("--tomo-alg", required=False, default = 'wbp', 
                        help="Tomogram algorithm used for produces segmentation prediction masks.")    
    parser.add_argument("--seg-info", type=parsers.parse_target, required=False, 
                        default='predict,octopi,1', help='Information Query to save Segmentation predictions under, e.g., (e.g., "name" or "name,user_id,session_id" - Default UserID is octopi and SessionID is 1')
    parser.add_argument("--tomo-batch-size", type=int, default=25, required=False, 
                        help="Batch size for tomogram processing.")
    parser.add_argument("--run-ids", type=parsers.parse_list, default=None, required=False, 
                        help="List of run IDs for prediction, e.g., run1,run2 or [run1,run2]. If not provided, all available runs will be processed.")   
    
def add_localize_parameters(parser):

    parser.add_argument("--voxel-size", type=int, required=False, default=10, help="Voxel size")
    parser.add_argument("--method", type=str,required=False, default='watershed', help="Localization method")
    parser.add_argument("--pick-session-id", required=False, default="1", type=str, help="Pick session ID")
    parser.add_argument("--pick-objects", required=True, type=str, help="Pick objects")
    parser.add_argument("--seg-info", required=True, type=str, help="Segmentation info")

def add_slurm_parameters(parser, base_job_name, gpus = 1):
    """
    Add SLURM job parameters to the parser.
    """
    parser.add_argument("--conda-env", type=str, required=False, default='/hpc/projects/group.czii/conda_environments/pyUNET/', help="Path to Conda environment")
    parser.add_argument("--job-name", type=str, required=False, default=f'{base_job_name}', help="Job name for SLURM job")        

    if gpus > 0:
        parser.add_argument("--gpu-constraint", type=str.lower, choices=['a6000', 'a100', 'h100', 'h200'], required=False, default='h100', help="GPU constraint")
    if gpus > 1:
        parser.add_argument("--num-gpus", type=int, required=False, default=1, help="Number of GPUs to use")
