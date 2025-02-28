from model_explore.submit_slurm import create_shellsubmit, create_multiconfig_shellsubmit
from model_explore.entry_points import run_train, run_segment_predict, run_localize
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
        output_file = 'train_model.log',
        shell_name = 'train_model.sh',
        conda_path = args.conda_env,
        command = command,
        num_gpus = 1,
        gpu_constraint = args.gpu_constraint
    )

def train_model_slurm():
    """
    Create a SLURM script for training 3D CNN models
    """
    parser_description = "Create a SLURM script for training 3D CNN models"
    args = run_train.train_model_parser(parser_description, add_slurm=True)
    create_train_script(args)   

def create_inference_script(args):
    
    if len(args.config.split(',')) > 1:

        create_multiconfig_shellsubmit(
            job_name = args.job_name,
            output_file = 'segment-predict.log',
            shell_name = 'segment.sh',
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
    --config {args.config} \\
    --seg-info  {",".join(args.seg_info)} \\
    --model-weights {args.model_weights} \\
    --dim-in {args.dim_in} --res-units {args.res_units} \\
    --model-type {args.model_type} --channels {",".join(map(str, args.channels))} --strides {",".join(map(str, args.strides))} \\
    --voxel-size {args.voxel_size} --tomo-algorithm {args.tomo_algorithm} --Nclass {args.Nclass}
"""

        create_shellsubmit(
            job_name = args.job_name,
            output_file = 'segment-predict.log',
            shell_name = 'segment.sh',
            conda_path = args.conda_env,
            command = command,
            num_gpus = 1,
            gpu_constraint = args.gpu_constraint
        )

def inference_slurm():
    
    parser_description = "Create a SLURM script for running segmentation predictions with a specified model and configuration on CryoET Tomograms."
    args = run_segment_predict.inference_parser(parser_description, add_slurm=True)
    create_inference_script(args)

def create_localize_script(args):
    
    if len(args.config.split(',')) > 1:
    
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
    --voxel-size {args.voxel_size} --pick-session-id {args.pick_session_id} --pick-user-id {args.pick_user_id} \\
    --method {args.method}  --seg-info {",".join(args.seg_info)} \\
"""
        if args.pick_objects is not None:
            command += f" --pick-objects {args.pick_objects}"

        create_shellsubmit(
            job_name = args.job_name,
            output_file = 'localize.log',
            shell_name = 'localize.sh',
            conda_path = args.conda_env,
            command = command,
            num_gpus = 0
        )

def localize_slurm():

    parser_description = "Create a SLURM script for localization on predicted segmentation masks"
    args = run_localize.localize_parser(parser_description, add_slurm=True)
    create_localize_script(args)
        