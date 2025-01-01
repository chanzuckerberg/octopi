from model_explore.submit_slurm import create_shellsubmit, create_multiconfig_shellsubmit
from model_explore import utils
import argparse

def create_train_script(args):

    strconfigs = ""
    for config in args.config:
        strconfigs += f"--config {config}"

    command = f"""
train-model \\
    --model-save-path {args.model_save_path} \\
    --voxel-size {args.voxel_size} --tomo-algorithm {args.tomo_algorithm} --Nclass {args.Nclass} \\
    --tomo-batch-size {args.tomo_batch_size} --num-epochs {args.num_epochs} --val-interval {args.val_interval} \\
    --target-name {args.target_name} --target-session-id {args.target_session_id} --target-user-id {args.target_user_id} \\
    --tversky-alpha {args.tversky_alpha} --model-type {args.model_type} --channels {args.channels} --strides {args.strides} --dim-in {args.dim_in} \\
    {strconfigs}
"""

    create_shellsubmit(
        job_name = args.job_name,
        output_file = args.output,
        shell_name = args.output_script,
        conda_path = args.conda_env,
        command = command,
        num_gpus = 1,
        gpu_constraint = args.gpu_constraint
    )

def create_inference_script(args):
    
    if len(args.config) > 1:

        create_multiconfig_shellsubmit(
            job_name = args.job_name,
            output_file = args.output,
            shell_name = args.output_script,
            conda_path = args.conda_env,
            base_inputs = args.base_inputs,
            config_inputs = args.config_inputs,
            command = args.command,
            num_gpus = args.num_gpus,
            gpu_constraint = args.gpu_constraint
        )
    else:

        command = f"""
inference \\
    --config {args.config[0]} \\
    --seg-info {args.seg_info} \\
    --model-weights {args.model_weights} \\
    --dim-in {args.dim_in} --res-units {args.res_units} \\
    --model-type {args.model_type} --channels {args.channels} --strides {args.strides} \\
    --voxel-size {args.voxel_size} --tomogram-algorithm {args.tomo_algorithm} --Nclass {args.Nclass}
"""

        create_shellsubmit(
            job_name = args.job_name,
            output_file = args.output,
            shell_name = args.output_script,
            conda_path = args.conda_env,
            command = command,
            num_gpus = args.num_gpus,
            gpu_constraint = args.gpu_constraint
        )

def create_localize_script(args):
    
    if len(args.config) > 1:
    
        create_multiconfig_shellsubmit(
            job_name = args.job_name,
            output_file = args.output,
            shell_name = args.output_script,
            conda_path = args.conda_env,
            base_inputs = args.base_inputs,
            config_inputs = args.config_inputs,
            command = args.command
        )
    else:

        command = f"""
localize \\
    --config {args.config} \\
    --voxel-size {args.voxel_size} \\
    --method {args.method} --pick-session-id {args.pick_session_id} --pick-objects {args.pick_objects} \\
    --seg-info {args.seg_info}
"""

        create_shellsubmit(
            job_name = args.job_name,
            output_file = args.output,
            shell_name = args.output_script,
            conda_path = args.conda_env,
            command = command,
            num_gpus = 0
        )

def add_model_parameters(parser):
    """
    Add common model parameters to the parser.
    """
    parser.add_argument("--config", type=str, required=True, action='append',
                            help="Specify a single configuration path (/path/to/config.json) "
                                "or multiple entries in the format session_name,/path/to/config.json. "
                                "Use multiple --config entries for multiple sessions.")
    parser.add_argument("--voxel-size", type=int, required=False, default=10, help="Voxel size of tomograms used")
    parser.add_argument("--model-type", type=str, required=False, default='UNet', help="Type of model: ['UNet', 'AttentionUNet']")
    parser.add_argument("--Nclass", type=int, required=False, default=3, help="Number of prediction classes in the model")
    parser.add_argument("--channels", type=str, required=False, default='32,64,128,128', help="List of channel sizes, e.g., 32,64,128,128")
    parser.add_argument("--strides", type=str, required=False, default='2,2,1', help="List of stride sizes, e.g., 2,2,1")
    parser.add_argument("--dim-in", type=int, required=False, default=96, help="Input dimension for the UNet model")
    parser.add_argument("--res-units", type=int, required=False, default=2, help="Number of residual units in the UNet")

def add_slurm_parameters(parser, base_job_name):
    """
    Add SLURM job parameters to the parser.
    """
    parser.add_argument("--conda-env", type=str, required=True, help="Path to Conda environment")
    parser.add_argument("--gpu-constraint", type=str, required=False, default='H100', help="GPU constraint")
    parser.add_argument("--output", type=str, required=False, default=f'{base_job_name}.log', help="Output log file for SLURM job")
    parser.add_argument("--output-script", type=str, required=False, default=f'{base_job_name}.sh', help="Name of SLURM shell script")
    parser.add_argument("--job-name", type=str, required=False, default=f'{base_job_name}', help="Job name for SLURM job")        

def cli():
    parser = argparse.ArgumentParser(description="SLURM script generator with sub-commands")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-command: train
    train_parser = subparsers.add_parser("train", help="Create a SLURM script for training 3D CNN models")
    add_model_parameters(train_parser)
    add_slurm_parameters(train_parser, 'train')
    train_parser.add_argument("--tomo-algorithm", required=False, default = 'wbp', help="Tomogram algorithm used for training")
    train_parser.add_argument("--target-name", required=True, help="Segmentation name used for training")
    train_parser.add_argument("--target-user-id", required=False, default = None, help="Segmentation UserID used for training")
    train_parser.add_argument("--target-session-id", required=False, default = None, help="Segmentation SessionID used for training")
    train_parser.add_argument("--model-save-path", required=True, help="Path to model save directory")
    train_parser.add_argument("--num-epochs", type=int, required=False, default = 100, help="Number of training epochs.")
    train_parser.add_argument("--val-interval", type=int, required=False, default=25, help="Interval for validation metric calculations.")
    train_parser.add_argument("--tomo-batch-size", type=int, required=False, default=15, help="Number of tomograms to load per epoch for training.")
    train_parser.add_argument("--tversky-alpha", type=float, required=False, default = 0.5, help="Alpha parameter for the Tversky loss.")
    train_parser.add_argument("--trainRunIDs", type=utils.parse_list, required=False, help="List of training run IDs, e.g., run1,run2,run3 or [run1,run2,run3].")
    train_parser.add_argument("--validateRunIDs", type=utils.parse_list, required=False, help="List of validation run IDs, e.g., run4,run5,run6 or [run4,run5,run6].")
    train_parser.set_defaults(func=create_train_script)

    # Sub-command: inference
    inference_parser = subparsers.add_parser("inference", help="Create a SLURM script for running inference on copick projects")
    add_model_parameters(inference_parser)
    add_slurm_parameters(inference_parser, 'inference')
    inference_parser.add_argument("--tomo-algorithm", required=False, default = 'wbp', help="Tomogram algorithm used for training")    
    inference_parser.add_argument("--seg-info", required=True, help='Information Query to save Segmentation predictions under, e.g., (e.g., "name" or "name,user_id,session_id" - Default UserID is DeepFindET and SessionID is 1')
    inference_parser.add_argument("--model-weights", required=True, help="Path to model weights")    
    inference_parser.add_argument("--num-gpus", type=int, required=False, default = 1, help="Number of GPUs to Request for Parallel Inference")
    inference_parser.set_defaults(func=create_inference_script)

    # Sub-command: localize
    localize_parser = subparsers.add_parser("localize", help="Create a SLURM script for localization on predicted segmentation masks")
    localize_parser.add_argument("--config", type=str, required=True, action='append',
                            help="Specify a single configuration path (/path/to/config.json) "
                                 "or multiple entries in the format session_name,/path/to/config.json. "
                                 "Use multiple --config entries for multiple sessions.")
    localize_parser.add_argument("--voxel-size", type=int, required=False, default=10, help="Voxel size")
    localize_parser.add_argument("--method", type=str,required=False, default='watershed', help="Localization method")
    localize_parser.add_argument("--pick-session-id", required=True, type=str, help="Pick session ID")
    localize_parser.add_argument("--pick-objects", required=True, type=str, help="Pick objects")
    localize_parser.add_argument("--seg-info", required=True, type=str, help="Segmentation info")
    add_slurm_parameters(localize_parser, 'localize')
    localize_parser.set_defaults(func=create_localize_script)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()