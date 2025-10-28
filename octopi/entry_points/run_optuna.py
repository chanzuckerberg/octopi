from octopi.pytorch.model_search_submitter import ModelSearchSubmit
from octopi.entry_points import common
from octopi.utils import parsers, io
import argparse, os, pprint

def optuna_parser(parser_description, add_slurm: bool = False):
    """
    Create an argument parser for model architecture search using Optuna.
    
    Args:
        parser_description (str): Description of the parser
        add_slurm (bool): Whether to add SLURM-specific arguments
    """
    
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input Arguments
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
    input_group.add_argument('--data-split', type=str, default='0.8', help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) "
                                "or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")

    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--model-type", type=str, default='Unet', required=False, 
                             choices=['Unet', 'AttentionUnet', 'MedNeXt', 'SegResNet'],
                             help="Model type to use for training")
    model_group.add_argument("--Nclass", type=int, default=3, required=False, help="Number of prediction classes in the model")

    train_group = parser.add_argument_group("Training Arguments")
    common.add_train_parameters(train_group, octopi = True)
    train_group.add_argument("--random-seed", type=int, default=42, required=False,
                             help="Random seed for reproducibility (default: 42).")

    # Worker mode arguments for distributed execution
    worker_group = parser.add_argument_group("Distributed Execution Arguments")
    worker_group.add_argument("--worker-mode", action="store_true", default=False,
                              help="Run as a worker that processes a single trial from an existing study.")
    worker_group.add_argument("--study-name", type=str, default=None,
                              help="Name of existing Optuna study to connect to (required for worker mode).")
    worker_group.add_argument("--study-db-path", type=str, default=None,
                              help="Path to SQLite database containing the study (required for worker mode).")
    worker_group.add_argument("--num-workers", type=int, default=None,
                              help="Number of SLURM worker jobs to submit (coordinator mode only).")

    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'optuna')

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    """
    CLI entry point for running optuna model architecture search.
    Supports both coordinator mode (creates study + submits workers) and worker mode (runs single trial).
    """

    description="Perform model architecture search with Optuna and MLflow integration."
    args = optuna_parser(description)

    # Parse the CoPick configuration paths
    if len(args.config) > 1:    copick_configs = parsers.parse_copick_configs(args.config)
    else:                       copick_configs = args.config[0]

    # Worker mode: load existing study and run a single trial
    if args.worker_mode:
        if not args.study_name or not args.study_db_path:
            raise ValueError("Worker mode requires --study-name and --study-db-path arguments.")

        print(f"Running in worker mode for study: {args.study_name}")
        print(f"Database path: {args.study_db_path}")

        search = ModelSearchSubmit(
            copick_config=copick_configs,
            target_name=args.target_info[0],
            target_user_id=args.target_info[1],
            target_session_id=args.target_info[2],
            tomo_algorithm=args.tomo_alg,
            voxel_size=args.voxel_size,
            Nclass=args.Nclass,
            model_type=args.model_type,
            mlflow_experiment_name=args.mlflow_experiment_name,
            random_seed=args.random_seed,
            num_epochs=args.num_epochs,
            num_trials=args.num_trials,
            trainRunIDs=args.trainRunIDs,
            validateRunIDs=args.validateRunIDs,
            tomo_batch_size=args.tomo_batch_size,
            best_metric=args.best_metric,
            val_interval=args.val_interval,
            data_split=args.data_split
        )

        # Run as worker: execute a single trial
        search.run_as_worker(study_name=args.study_name, db_path=args.study_db_path)
        return

    # Coordinator mode: create study and optionally submit workers
    # Create the model exploration directory
    os.makedirs(f'explore_results_{args.model_type}', exist_ok=True)

    # Save parameters
    save_parameters(args, f'explore_results_{args.model_type}/octopi.yaml')

    # Call the function with parsed arguments
    search = ModelSearchSubmit(
        copick_config=copick_configs,
        target_name=args.target_info[0],
        target_user_id=args.target_info[1],
        target_session_id=args.target_info[2],
        tomo_algorithm=args.tomo_alg,
        voxel_size=args.voxel_size,
        Nclass=args.Nclass,
        model_type=args.model_type,
        mlflow_experiment_name=args.mlflow_experiment_name,
        random_seed=args.random_seed,
        num_epochs=args.num_epochs,
        num_trials=args.num_trials,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
        tomo_batch_size=args.tomo_batch_size,
        best_metric=args.best_metric,
        val_interval=args.val_interval,
        data_split=args.data_split
    )

    # If num_workers is specified, submit distributed workers
    if args.num_workers:
        print(f"\nCoordinator mode: Submitting {args.num_workers} worker jobs...")
        search.submit_workers(args=args, num_workers=args.num_workers)
    else:
        # Run the model search locally (original behavior)
        search.run_model_search()

def save_parameters(args: argparse.Namespace, 
                    output_path: str):
    """
    Save the Optuna search parameters to a JSON file.
    Args:
        args: Parsed arguments from argparse.
        output_path: Path to save the JSON file.
    """
    # Organize parameters into categories
    params = {
        "input": {
            "copick_config": args.config,
            "target_info": args.target_info,
            "tomo_algorithm": args.tomo_alg,
            "voxel_size": args.voxel_size,
            "Nclass": args.Nclass,            
        },
        "optimization": {
            "model_type": args.model_type,
            "mlflow_experiment_name": args.mlflow_experiment_name,
            "random_seed": args.random_seed,
            "num_trials": args.num_trials,
            "best_metric": args.best_metric
        },
        "training": {
            "num_epochs": args.num_epochs,            
            "tomo_batch_size": args.tomo_batch_size,
            "trainRunIDs": args.trainRunIDs,
            "validateRunIDs": args.validateRunIDs,
            "data_split": args.data_split
        }
    }

    # Print the parameters
    print(f"\nParameters for Model Architecture Search:")
    pprint.pprint(params); print()

    # Save to YAML file
    io.save_parameters_yaml(params, output_path)

if __name__ == "__main__":
    cli()