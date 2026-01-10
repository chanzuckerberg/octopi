from octopi.utils import parsers
import rich_click as click
from typing import List, Tuple

def model_parameters(octopi: bool = False):
    """Decorator for adding model parameters"""
    def decorator(f):
        f = click.option("-dim", "--dim-in", type=int, default=96, 
                        help="Input dimension for the UNet model")(f)
        f = click.option("--res-units", type=int, default=1,
                        help="Number of residual units in the UNet")(f)
        f = click.option("-s", "--strides", type=str, default='2,2,1',
                        callback=lambda ctx, param, value: parsers.parse_int_list(value) if value else value,
                        help="List of stride sizes")(f)
        f = click.option("-ch", "--channels", type=str, default='32,64,96,96',
                        callback=lambda ctx, param, value: parsers.parse_int_list(value) if value else value,
                        help="List of channel sizes")(f)
        return f
    return decorator

def inference_model_parameters():
    """Decorator for adding inference model parameters"""
    def decorator(f):
        f = click.option("-mw", "--model-weights", type=click.Path(exists=True), required=True,
                        help="Path to the model weights file")(f)
        f = click.option("-mc", "--model-config", type=click.Path(exists=True), required=True,
                        help="Path to the model configuration file")(f)
        return f
    return decorator

def train_parameters(octopi: bool = False):
    """Decorator for adding training parameters"""
    def decorator(f):
        if octopi:
            f = click.option("-nt", "--num-trials", type=int, default=10,
                            help="Number of trials for architecture search")(f)
        else:
            f = click.option("-o", "--model-save-path", type=click.Path(), default='results',
                            help="Path to model save directory")(f)
            f = click.option("--tversky-alpha", type=float, default=0.3,
                            help="Alpha parameter for the Tversky loss")(f)
            f = click.option("-lr", "--lr", type=float, default=1e-3,
                            help="Learning rate for the optimizer")(f)
            f = click.option('-bs', "--batch-size", type=int, default=16,
                            help="Batch size for training")(f)
        
        f = click.option("--best-metric", type=str, default='avg_f1',
                        help="Metric to Monitor for Determining Best Model. To track fBetaN, use fBetaN with N as the beta-value.")(f)
        cdf = click.option('-ncache', "--ncache-tomos", type=int, default=15,
                        help="Number of tomograms kept in memory and used for training in each epoch (SmartCache window size).")(f)
        f = click.option("--val-interval", type=int, default=10,
                        help="Interval for validation metric calculations")(f)
        f = click.option('-nepochs', "--num-epochs", type=int, default=1000,
                        help="Number of training epochs")(f)
        return f
    return decorator

def config_parameters(single_config: bool):
    """Decorator for adding config parameters"""
    def decorator(f):
        f = click.option("-vs", "--voxel-size", type=float, default=10,
                        help="Voxel size of tomograms used")(f)
        if single_config:
            f = click.option("-c", "--config", type=click.Path(exists=True), required=True,
                            help="Path to the configuration file")(f)
        else:
            f = click.option("-c", "--config", type=str, multiple=True, required=True,
                            help="Specify a single configuration path (/path/to/config.json) or multiple entries in the format session_name,/path/to/config.json. Use multiple --config entries for multiple sessions.")(f)
        return f
    return decorator

def inference_parameters():
    """Decorator for adding inference parameters"""
    def decorator(f):
        f = click.option('-runs', "--run-ids", type=str, default=None,
                        callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
                        help="List of run IDs for prediction, e.g., run1,run2 or [run1,run2]. If not provided, all available runs will be processed.")(f)
        f = click.option('-ntomos', "--tomo-batch-size", type=int, default=1,
                        help="Batch size for tomogram processing")(f)
        f = click.option('-seginfo', "--seg-info", type=str, default='predict,octopi,1',
                        callback=lambda ctx, param, value: parsers.parse_target(value) if value else value,
                        help='Information Query to save Segmentation predictions under (e.g., "name" or "name,user_id,session_id" - Default UserID is octopi and SessionID is 1')(f)
        f = click.option('-alg', "--tomo-alg", type=str, default='wbp',
                        help="Tomogram algorithm used for produces segmentation prediction masks")(f)
        return f
    return decorator

def localize_parameters():
    """Decorator for adding localization parameters"""
    def decorator(f):
        f = click.option('-seginfo', "--seg-info", type=str, required=True,
                        help="Segmentation info")(f)
        f = click.option("--pick-objects", type=str, required=True,
                        help="Pick objects")(f)
        f = click.option("--pick-session-id", type=str, default="1",
                        help="Pick session ID")(f)
        f = click.option('-m', "--method", type=str, default='watershed',
                        help="Localization method")(f)
        f = click.option('-vs', "--voxel-size", type=int, default=10,
                        help="Voxel size")(f)
        return f
    return decorator

def slurm_parameters(base_job_name: str, gpus: int = 1):
    """Decorator for adding SLURM parameters"""
    def decorator(f):
        if gpus > 1:
            f = click.option("--num-gpus", type=int, default=1,
                            help="Number of GPUs to use")(f)
        if gpus > 0:
            f = click.option("--gpu-constraint", type=click.Choice(['a6000', 'a100', 'h100', 'h200'], case_sensitive=False),
                            default='h100',
                            help="GPU constraint")(f)
        f = click.option("--job-name", type=str, default=base_job_name,
                        help="Job name for SLURM job")(f)
        f = click.option("--conda-env", type=click.Path(), default='/hpc/projects/group.czii/conda_environments/pyUNET/',
                        help="Path to Conda environment")(f)
        return f
    return decorator