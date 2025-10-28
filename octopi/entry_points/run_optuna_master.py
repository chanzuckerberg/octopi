"""
Entry point for running Optuna model search with a master monitoring job.

The master job creates the Optuna study and continuously submits/monitors
worker jobs that each execute a single trial. This provides better scheduling
flexibility and resource utilization on SLURM clusters.
"""

from octopi.entry_points import run_optuna, common
from octopi.utils import parsers
import argparse
import os


def optuna_master_parser(parser_description: str, add_slurm: bool = False):
    """
    Create an argument parser for master-managed model architecture search.

    Args:
        parser_description (str): Description of the parser
        add_slurm (bool): Whether to add SLURM-specific arguments
    """

    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input Arguments (same as regular optuna)
    input_group = parser.add_argument_group("Input Arguments")
    common.add_config(input_group, single_config=False)
    input_group.add_argument("--target-info", type=parsers.parse_target, default="targets,octopi,1",
                             help="Target information, e.g., 'name' or 'name,user_id,session_id'")
    input_group.add_argument("--tomo-alg", default='wbp',
                             help="Tomogram algorithm used for training")
    input_group.add_argument("--mlflow-experiment-name", type=str, default="model-search", required=False,
                             help="Name of the MLflow experiment (default: 'model-search').")
    input_group.add_argument("--trainRunIDs", type=parsers.parse_list, default=None, required=False,
                             help="List of training run IDs, e.g., run1,run2 or [run1,run2].")
    input_group.add_argument("--validateRunIDs", type=parsers.parse_list, default=None, required=False,
                             help="List of validation run IDs, e.g., run3,run4 or [run3,run4].")
    input_group.add_argument('--data-split', type=str, default='0.8',
                             help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) "
                                  "or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")

    # Model Arguments
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--model-type", type=str, default='Unet', required=False,
                             choices=['Unet', 'AttentionUnet', 'MedNeXt', 'SegResNet'],
                             help="Model type to use for training")

    # Training Arguments
    train_group = parser.add_argument_group("Training Arguments")
    common.add_train_parameters(train_group, octopi=True)
    train_group.add_argument("--random-seed", type=int, default=42, required=False,
                             help="Random seed for reproducibility (default: 42).")

    # Master Job Configuration
    master_group = parser.add_argument_group("Master Job Configuration")
    master_group.add_argument("--max-parallel-workers", type=int, default=5, required=False,
                              help="Maximum number of workers to run in parallel (default: 5).")
    master_group.add_argument("--worker-time-limit", type=str, default="08:00:00", required=False,
                              help="Time limit for each worker job in HH:MM:SS format (default: '08:00:00').")
    master_group.add_argument("--master-time-limit", type=str, default="72:00:00", required=False,
                              help="Time limit for the master monitoring job in HH:MM:SS format (default: '72:00:00').")
    master_group.add_argument("--monitor-interval", type=int, default=300, required=False,
                              help="Seconds between progress checks and worker submissions (default: 300).")
    master_group.add_argument("--max-worker-retries", type=int, default=2, required=False,
                              help="Maximum number of times to retry a failed worker (default: 2).")
    master_group.add_argument("--error-threshold", type=float, default=0.2, required=False,
                              help="Maximum failure rate (failed/total) before aborting (default: 0.2).")

    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'optuna-master')

    args = parser.parse_args()
    return args


def cli():
    """
    CLI entry point for master-managed optuna model architecture search.

    This command creates a SLURM submission script for a master monitoring job.
    The master job will:
    1. Create the Optuna study
    2. Submit worker jobs dynamically
    3. Monitor progress and handle failures
    4. Terminate when all trials are complete
    """

    description = (
        "Create a master SLURM job for Optuna model architecture search. "
        "The master job monitors progress and dynamically submits worker jobs."
    )
    args = optuna_master_parser(description, add_slurm=True)

    # Import here to avoid circular dependency
    from octopi.entry_points.create_slurm_submission import create_master_optuna_script

    # Create the SLURM submission script
    create_master_optuna_script(args)

    print("\n" + "=" * 70)
    print("Master job script created successfully!")
    print("=" * 70)
    print(f"\nTo submit the job:")
    print(f"  sbatch model_explore_master.sh")
    print(f"\nTo monitor progress:")
    print(f"  tail -f explore_results_{args.model_type}/master.log")
    print(f"\nTo stop gracefully:")
    print(f"  touch .stop_optuna_master")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    cli()
