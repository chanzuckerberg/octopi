from octopi.entry_points import run_train, run_segment_predict, run_localize, run_optuna
from octopi.utils.submit_slurm import create_shellsubmit, create_multiconfig_shellsubmit
from octopi.processing.importers import cli_mrcs_parser, cli_dataportal_parser
from octopi.entry_points import common 
from octopi import utils
import argparse

def create_train_script(args):
    """
    Create a SLURM script for training 3D CNN models
    """

    strconfigs = ""
    for config in args.config:
        strconfigs += f"--config {config}"

    command = f"""
octopi train \\
    {strconfigs} \\
    --model-save-path {args.model_save_path} \\
    --target-info {','.join(args.target_info)} \\
    --voxel-size {args.voxel_size} --tomo-alg {args.tomo_alg} --Nclass {args.Nclass} \\
    --tomo-batch-size {args.tomo_batch_size} --num-tomo-crops {args.num_tomo_crops} \\
    --best-metric {args.best_metric} --num-epochs {args.num_epochs} --val-interval {args.val_interval} \\
    """

    # If a model config is provided, use it to build the model
    if args.model_config is not None:
        command += f" --model-config {args.model_config}"
    else:
        channels = ",".join(map(str, args.channels))
        strides = ",".join(map(str, args.strides))
        command += (
            f" --tversky-alpha {args.tversky_alpha}"
            f" --channels {channels}"
            f" --strides {strides}"
            f" --dim-in {args.dim_in}"
            f" --res-units {args.res_units}"
        )

    # If Model Weights are provided, use them to initialize the model
    if args.model_weights is not None and args.model_config is not None:
        command += f" --model-weights {args.model_weights}"

    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    create_shellsubmit(
        job_name = args.job_name,
        output_file = 'trainer.log',
        shell_name = 'train_octopi.sh',
        command = command,
        slurm_config = slurm_config,
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

def create_model_explore_script(args):
    """
    Create a SLURM script for running bayesian optimization on 3D CNN models
    """
    strconfigs = ""
    for config in args.config:
        strconfigs += f"--config {config}"

    command = f"""
octopi model-explore \\
    --model-type {args.model_type} --model-save-path {args.model_save_path} \\
    --voxel-size {args.voxel_size} --tomo-alg {args.tomo_alg} --Nclass {args.Nclass} \\
    --val-interval {args.val_interval} --num-epochs {args.num_epochs} --num-trials {args.num_trials} \\
    --best-metric {args.best_metric} --mlflow-experiment-name {args.mlflow_experiment_name} \\
    --target-name {args.target_name} --target-session-id {args.target_session_id} --target-user-id {args.target_user_id} \\
    {strconfigs}
"""

    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    create_shellsubmit(
        job_name = args.job_name,
        output_file = 'optuna.log',
        shell_name = 'model_explore.sh',
        command = command,
        slurm_config = slurm_config,
        num_gpus = 1,
        gpu_constraint = args.gpu_constraint
    )

def model_explore_slurm():
    """
    Create a SLURM script for running bayesian optimization on 3D CNN models
    """
    parser_description = "Create a SLURM script for running bayesian optimization on 3D CNN models"
    args = run_optuna.optuna_parser(parser_description, add_slurm=True)
    create_model_explore_script(args)

def create_master_optuna_script(args):
    """
    Create a SLURM script for the master monitoring job.

    The master job runs on CPU and monitors Optuna progress while dynamically
    submitting GPU worker jobs.
    """
    # Build the command to run the master loop
    cmd_parts = ['octopi', '_run-master-loop']

    # Add all configuration arguments
    for config in args.config:
        cmd_parts.extend(['--config', config])

    cmd_parts.extend(['--target-info', f"{args.target_info[0]},{args.target_info[1]},{args.target_info[2]}"])
    cmd_parts.extend(['--tomo-alg', args.tomo_alg])
    cmd_parts.extend(['--model-type', args.model_type])
    cmd_parts.extend(['--Nclass', str(args.Nclass)])
    cmd_parts.extend(['--voxel-size', str(args.voxel_size)])
    cmd_parts.extend(['--num-epochs', str(args.num_epochs)])
    cmd_parts.extend(['--num-trials', str(args.num_trials)])
    cmd_parts.extend(['--tomo-batch-size', str(args.tomo_batch_size)])
    cmd_parts.extend(['--best-metric', args.best_metric])
    cmd_parts.extend(['--val-interval', str(args.val_interval)])
    cmd_parts.extend(['--mlflow-experiment-name', args.mlflow_experiment_name])
    cmd_parts.extend(['--random-seed', str(args.random_seed)])
    cmd_parts.extend(['--data-split', args.data_split])

    # Add master-specific arguments
    cmd_parts.extend(['--max-parallel-workers', str(args.max_parallel_workers)])
    cmd_parts.extend(['--worker-time-limit', args.worker_time_limit])
    cmd_parts.extend(['--master-time-limit', args.master_time_limit])
    cmd_parts.extend(['--monitor-interval', str(args.monitor_interval)])
    cmd_parts.extend(['--max-worker-retries', str(args.max_worker_retries)])
    cmd_parts.extend(['--error-threshold', str(args.error_threshold)])

    # Add optional run IDs if provided
    if args.trainRunIDs:
        cmd_parts.extend(['--trainRunIDs', ','.join(args.trainRunIDs)])
    if args.validateRunIDs:
        cmd_parts.extend(['--validateRunIDs', ','.join(args.validateRunIDs)])

    command = ' '.join(cmd_parts)

    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    # Create the SLURM script for master job (CPU-only, long-running)
    create_shellsubmit(
        job_name=args.job_name,
        output_file=f'explore_results_{args.model_type}/master_slurm.log',
        shell_name='model_explore_master.sh',
        command=command,
        slurm_config=slurm_config,
        num_gpus=0,  # Master runs on CPU
        time_limit=args.master_time_limit,
        cpus_per_task=4,
        mem_per_cpu='16G',
        partition='cpu'
    )

    print(f"\nMaster job configuration:")
    print(f"  Max parallel workers: {args.max_parallel_workers}")
    print(f"  Worker time limit: {args.worker_time_limit}")
    print(f"  Master time limit: {args.master_time_limit}")
    print(f"  Monitor interval: {args.monitor_interval}s")
    print(f"  Max worker retries: {args.max_worker_retries}")

def model_explore_master_slurm():
    """
    Create a SLURM script for master-managed bayesian optimization.

    The master job monitors progress and dynamically submits worker jobs.
    """
    from octopi.entry_points import run_optuna_master

    parser_description = "Create a SLURM script for master-managed bayesian optimization on 3D CNN models"
    args = run_optuna_master.optuna_master_parser(parser_description, add_slurm=True)
    create_master_optuna_script(args)   

def create_inference_script(args):
    """
    Create a SLURM script for running inference on 3D CNN models
    """
    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    if len(args.config.split(',')) > 1:

        create_multiconfig_shellsubmit(
            job_name = args.job_name,
            output_file = 'predict.log',
            shell_name = 'segment.sh',
            base_inputs = args.base_inputs,
            config_inputs = args.config_inputs,
            command = args.command,
            slurm_config = slurm_config
        )
    else:

        command = f"""
octopi inference \\
    --config {args.config} \\
    --seg-info  {",".join(args.seg_info)} \\
    --model-weights {args.model_weights} \\
    --dim-in {args.dim_in} --res-units {args.res_units} \\
    --model-type {args.model_type} --channels {",".join(map(str, args.channels))} --strides {",".join(map(str, args.strides))} \\
    --voxel-size {args.voxel_size} --tomo-alg {args.tomo_alg} --Nclass {args.Nclass}
"""

        create_shellsubmit(
            job_name = args.job_name,
            output_file = 'predict.log',
            shell_name = 'segment.sh',
            command = command,
            slurm_config = slurm_config,
            num_gpus = 1,
            gpu_constraint = args.gpu_constraint
        )

def inference_slurm():
    """
    Create a SLURM script for running segmentation predictions with a specified model and configuration on CryoET Tomograms.
    """
    parser_description = "Create a SLURM script for running segmentation predictions with a specified model and configuration on CryoET Tomograms."
    args = run_segment_predict.inference_parser(parser_description, add_slurm=True)
    create_inference_script(args)

def create_localize_script(args):
    """"
    Create a SLURM script for running localization on predicted segmentation masks
    """
    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    if len(args.config.split(',')) > 1:

        create_multiconfig_shellsubmit(
            job_name = args.job_name,
            output_file = args.output,
            shell_name = args.output_script,
            base_inputs = args.base_inputs,
            config_inputs = args.config_inputs,
            command = args.command,
            slurm_config = slurm_config
        )
    else:

        command = f"""
octopi localize \\
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
            command = command,
            slurm_config = slurm_config,
            num_gpus = 0
        )

def localize_slurm():
    """
    Create a SLURM script for running localization on predicted segmentation masks
    """
    parser_description = "Create a SLURM script for localization on predicted segmentation masks"
    args = run_localize.localize_parser(parser_description, add_slurm=True)
    create_localize_script(args)
        
def create_extract_mb_picks_script(args):
    pass

def extract_mb_picks_slurm():
    pass


def create_import_mrc_script(args):
    """
    Create a SLURM script for importing mrc volumes and potentialy downsampling
    """
    command = f"""
octopi import-mrc-volumes \\
    --mrcs-path {args.mrcs_path} \\
    --config {args.config} --target-tomo-type {args.target_tomo_type} \\
    --input-voxel-size {args.input_voxel_size} --output-voxel-size {args.output_voxel_size}
"""

    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    create_shellsubmit(
        job_name = args.job_name,
        output_file = 'importer.log',
        shell_name = 'mrc_importer.sh',
        command = command,
        slurm_config = slurm_config
    )

def import_mrc_slurm():
    """
    Create a SLURM script for importing mrc volumes and potentialy downsampling
    """
    parser_description = "Create a SLURM script for importing mrc volumes and potentialy downsampling"
    args = cli_mrcs_parser(parser_description, add_slurm=True)
    create_import_mrc_script(args)


def create_download_dataportal_script(args):
    """
    Create a SLURM script for downloading tomograms from the Dataportal
    """
    command = f"""
octopi download-dataportal \\
    --config {args.config} --datasetID {args.datasetID} \\
    --overlay-path {args.overlay_path}
    --dataportal-name {args.dataportal_name} --target-tomo-type {args.target_tomo_type} \\
    --input-voxel-size {args.input_voxel_size} --output-voxel-size {args.output_voxel_size}
"""

    # Load SLURM configuration
    from octopi.utils.slurm_config import SLURMConfig
    slurm_config = SLURMConfig.load()
    slurm_config.apply_cli_overrides(args)

    create_shellsubmit(
        job_name = args.job_name,
        output_file = 'importer.log',
        shell_name = 'dataportal_importer.sh',
        command = command,
        slurm_config = slurm_config
    )

def download_dataportal_slurm():
    """
    Create a SLURM script for downloading tomograms from the Dataportal
    """
    parser_description = "Create a SLURM script for downloading tomograms from the Dataportal"
    args = cli_dataportal_parser(parser_description, add_slurm=True)
    create_download_dataportal_script(args)
