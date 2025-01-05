from model_explore.submit_slurm import create_shellsubmit, create_multiconfig_shellsubmit
from model_explore.entry_points import common 
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

def train_model_slurm():
    parser = argparse.ArgumentParser(
        description="Create a SLURM script for training 3D CNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    common.add_config(parser, single_config=False)
    common.add_model_parameters(parser)
    common.add_train_parameters(parser)
    parser.add_argument("--tomo-algorithm", default='wbp', help="Tomogram algorithm used for training")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, help="List of training run IDs, e.g., run1,run2,run3")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, help="List of validation run IDs, e.g., run4,run5,run6")
    parser.add_argument("--mlflow", type=utils.string2bool, default=False, help="Log the results with MLFlow or save in a unique directory")
    common.add_slurm_parameters(parser, 'train')

    args = parser.parse_args()
    create_train_script(args)   

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

def inference_slurm():
    parser = argparse.ArgumentParser(
        description="Create a SLURM script for training 3D CNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    common.add_config(parser, single_config=True)
    common.add_model_parameters(parser)
    common.add_slurm_parameters(parser, 'inference')
    common.add_inference_parameters(parser)
    parser.add_argument("--num-gpus", type=int, required=False, default = 1, 
                                  help="Number of GPUs to Request for Parallel Inference")
    parser.set_defaults(func=create_inference_script)
    args = parser.parse_args()
    create_inference_script(args)

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

def localize_slurm():
    parser = argparse.ArgumentParser(
        description="Create a SLURM script for localization on predicted segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    common.add_config(parser, single_config=True)
    common.add_localize_parameters(parser)
    common.add_slurm_parameters(parser, 'localize')
    parser.set_defaults(func=create_localize_script)
    args = parser.parse_args()
    create_localize_script(args)
        
