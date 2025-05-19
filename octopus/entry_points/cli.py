from octopus.processing.download import cli as download_dataportal
from octopus.entry_points.run_create_targets import cli as create_targets
from octopus.entry_points.run_train import cli as train_model
from octopus.entry_points.run_optuna import cli as model_explore
from octopus.entry_points.run_segment_predict import cli as inference
from octopus.entry_points.run_localize import cli as localize
from octopus.entry_points.run_evaluate import cli as evaluate
from octopus.entry_points.run_extract_mb_picks import cli as extract_mb_picks
import octopus.entry_points.create_slurm_submission as slurm_submitter
import argparse
import sys

def cli_main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Octopus 🐙: 🛠️ Tools for Finding Proteins in 🧊 cryo-ET data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True  # Make subcommand required

    # Define all subcommands with their help text
    commands = {
        "download-dataportal": (download_dataportal, "Download tomograms from the Dataportal, we can downsample to smaller voxel size if desired."),
        "create-targets": (create_targets, "Generate segmentation targets from coordinates."),
        "train": (train_model, "Train a single U-Net model."),
        "model-explore": (model_explore, "Explore model architectures with Optuna / Bayesian Optimization."),
        "inference": (inference, "Perform segmentation inference on tomograms."),
        "localize": (localize, "Perform localization of particles in tomograms."),
        "extract-mb-picks": (extract_mb_picks, "Extract MB Picks from tomograms."),
        "evaluate": (evaluate, "Evaluate the performance of a model."),
    }

    # Add all subparsers and their help text
    for cmd_name, (cmd_func, cmd_help) in commands.items():
        subparsers.add_parser(cmd_name, help=cmd_help)

    # Parse just the command part to determine which subcommand was chosen
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv[1]
        cmd_func = commands[command][0]
        
        # Remove the first argument (command name) and call the appropriate CLI function
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cmd_func()
    else:
        # Just show help if no valid command
        parser.parse_args()

def cli_slurm_main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Octopus for SLURM 🖥️: Shell Submission Tools for Running 🐙 on HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True  # Make subcommand required

    # Define all subcommands with their help text
    commands = {
        # "download-dataportal": (download_dataportal, "Download tomograms from the Dataportal, we can downsample to smaller voxel size if desired."),
        # "create-targets": (create_targets, "Generate segmentation targets from coordinates."),
        "train": (slurm_submitter.train_model_slurm, "Train a single U-Net model."),
        "model-explore": (slurm_submitter.model_explore_slurm, "Explore model architectures with Optuna / Bayesian Optimization."),
        "inference": (slurm_submitter.inference_slurm, "Perform segmentation inference on tomograms."),
        "localize": (slurm_submitter.localize_slurm, "Perform localization of particles in tomograms."),
        # "extract-mb-picks": (extract_mb_picks, "Extract MB Picks from tomograms.")
        # "evaluate": (evaluate, "Evaluate the performance of a model."),
    }

    # Add all subparsers and their help text
    for cmd_name, (cmd_func, cmd_help) in commands.items():
        subparsers.add_parser(cmd_name, help=cmd_help)

    # Parse just the command part to determine which subcommand was chosen
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv[1]
        cmd_func = commands[command][0]
        
        # Remove the first argument (command name) and call the appropriate CLI function
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cmd_func()
    else:
        # Just show help if no valid command
        parser.parse_args()
