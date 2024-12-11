from model_explore.pytorch.datasets import generators, multi_config_generator
from model_explore.pytorch import io, hyper_search, utils
import torch, mlflow, optuna, argparse, json, os, pprint
from typing import List, Optional

def model_search(
    copick_config: str,
    target_name: str,
    target_user_id: str,
    target_session_id: str,
    tomo_algorithm: str,
    voxel_size: float,    
    Nclass: int,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str, 
    random_seed: int,
    num_epochs: int,
    num_trials: int,
    tomo_batch_size: int,
    best_metric: str,
    trainRunIDs: List[str],
    validateRunIDs: List[str],   
    ):
    """
    Perform model architecture search using Optuna, MLflow, and a custom data generator.

    Parameters:
    - copick_config (str): Path to CoPick configuration file.
    - target_name (str): Name of the target for segmentation.
    - target_user_id (str): Optional user ID for tracking (default: None).
    - target_session_id (str): Optional session ID for tracking (default: None).
    - tomo_algorithm (str): Tomogram algorithm to use (default: 'wbp').
    - voxel_size (float): Voxel size for tomograms (default: 10).
    - Nclass (int): Number of prediction classes (default: 3).
    - mlflow_experiment_name (str): MLflow experiment name (default: 'model-search').
    - mlflow_tracking_uri (str): URI for MLflow tracking server.
    - random_seed (int): Seed for reproducibility (default: 42).
    - num_epochs (int): Number of epochs per trial.
    - num_trials (int): Number of trials for hyperparameter optimization.
    - tomo_batch_size (int): Batch size for tomogram loading.
    - trainRunIDs (List[str]): List of training run IDs.
    - validateRunIDs (List[str]): List of validation run IDs.
    """

    # Random Seed
    utils.set_seed(random_seed)

    # Initialize the data generator to manage training and validation datasets
    print_input_configs(copick_config)
    if isinstance(copick_config, dict):
        # Multi-config training
        data_generator = multi_config_generator.MultiConfigTrainLoaderManager(
            copick_config, 
            target_name, 
            target_session_id = target_session_id,
            target_user_id = target_user_id,
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = Nclass,
            tomo_batch_size = tomo_batch_size )
    else:
        # Single-config training
        data_generator = generators.TrainLoaderManager(
            copick_config, 
            target_name, 
            target_session_id = target_session_id,
            target_user_id = target_user_id,
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = Nclass,
            tomo_batch_size = tomo_batch_size ) 

    # Split datasets into training and validation
    data_generator.get_data_splits(trainRunIDs = trainRunIDs,
                                   validateRunIDs = validateRunIDs)

    import pdb; podb.set_trace()

    # Get the reload frequency
    data_generator.get_reload_frequency(num_epochs) 

    # Define Optuna pruning strategy to stop unpromising trials
    pruning = True
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    )

    # Option 1: TPE sampler - Tree-structured Parzen Estimator (TPE) for hyperparameter sampling
    tpe_sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,     # Number of initial random trials before TPE kicks in
        n_ei_candidates=24,      # Number of candidate samples for Expected Improvement
        multivariate=True        # Use multivariate TPE for correlated parameter sampling
    )

    # # Option 2: BoTorchSampler
    # botorch_sampler = optuna.samplers.BoTorchSampler(
    #     n_startup_trials=10,     # Number of initial random trials before TPE kicks in
    #     multivariate=True        # Use multivariate TPE for correlated parameter sampling
    # )

    # Create Save Folder if It Doesn't Exist
    os.makedirs('model_exploration', exist_ok=True)      

    # Initialize MLflow for experiment tracking
    try:
        utils.mlflow_setup()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except:
        pass
    mlflow.set_experiment(mlflow_experiment_name)    

    print(f'Running Architecture Search Over 1 GPU\n')
    storage = "sqlite:///trials.db"

    # Get available GPUs (for example, on an 4 GPU node) - Run the appropriate function based on the number of GPUs
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        multi_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, 
                         random_seed, num_trials, best_metric, gpu_count)
    else:
        single_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, 
                          random_seed, num_trials, best_metric)
    

def single_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, random_seed, num_trials, best_metric):
    """
    Run Optuna optimization on a single GPU.
    """
    with mlflow.start_run():

        study = optuna.create_study(storage=storage,
                                    direction="maximize",
                                    sampler=tpe_sampler,
                                    load_if_exists=True,
                                    pruner=pruner)

        mlflow.log_params({"random_seed": random_seed})
        mlflow.log_params(data_generator.get_dataloader_parameters())

        # Determine Device to Run Optuna On
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        study.optimize(lambda trial: hyper_search.objective(trial, num_epochs, device, 
                                                            data_generator, random_seed = random_seed, 
                                                            best_metric = best_metric), 
                        n_trials=num_trials)

        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_params}")
    
def multi_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, random_seed, num_trials, best_metric, gpu_count):
    """
    Run Optuna optimization on multiple GPUs.
    """     
    with mlflow.start_run() as parent_run:

        study = optuna.create_study(storage=storage,
                                    direction="maximize",
                                    sampler=tpe_sampler,
                                    load_if_exists=True,
                                    pruner=pruner)

        mlflow.log_params({"random_seed": random_seed})
        mlflow.log_params(data_generator.get_dataloader_parameters())
        
        parent_run_id = parent_run.info.run_id    
        study.optimize(lambda trial: hyper_search.multi_gpu_objective(parent_run_id, trial, 
                                                                      num_epochs,
                                                                      data_generator, 
                                                                      best_metric = best_metric,
                                                                      gpu_count = gpu_count), 
                       n_trials=num_trials,
                       n_jobs=gpu_count) # Run trials on multiple GPUs

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")

def print_input_configs(copick_config):

    print(f'\nTraining with:')
    if isinstance(copick_config, dict):
        for session, config in copick_config.items():
            print(f'  {session}: {config}')
    else:
        print(f'  {copick_config}')
    print()

# Entry point with argparse
def cli():
    """
    CLI entry point for running optuna model archetecture search.
    """
    parser = argparse.ArgumentParser(description="Perform model architecture search with Optuna and MLflow integration.")

    # Required arguments
    # parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--config", type=str, required=True, action='append',
                            help="Specify a single configuration path (/path/to/config.json) "
                                 "or multiple entries in the format session_name,/path/to/config.json. "
                                 "Use multiple --config entries for multiple sessions.")    
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target to segment.")

    # Optional arguments
    parser.add_argument("--target-user-id", type=str, required=False, default=None, help="User ID of the target.")
    parser.add_argument("--target-session-id", type=str, required=False, default=None, help="Session ID of the target.")
    parser.add_argument("--tomo-algorithm", type=str, default='wbp', required=False, help="Tomogram algorithm to use (default: 'wbp').")
    parser.add_argument("--voxel-size", type=float, default=10, required=False, help="Voxel size for tomograms (default: 10).")    
    parser.add_argument("--Nclass", type=int, default=3, required=False, help="Number of classes for prediction.")
    parser.add_argument("--mlflow-experiment-name", type=str, default="model-search", required=False, help="Name of the MLflow experiment (default: 'model-search').")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://mlflow.mlflow.svc.cluster.local:5000", required=False, help="URI for the MLflow tracking server (default: 'http://mlflow.mlflow.svc.cluster.local:5000').")
    parser.add_argument("--random-seed", type=int, default=42, required=False, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num-epochs", type=int, default=100, required=False, help="Number of epochs per trial (default: 100).")
    parser.add_argument("--num-trials", type=int, default=10, required=False, help="Number of trials for architecture search (default: 10).")
    parser.add_argument("--tomo-batch-size", type=int, default=20, required=False, help="Batch size for tomograms (default: 20).")
    parser.add_argument("--best-metric", type=str, default='avg_f1', required=False, help="Metric to Monitor for Optimization")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, default=None, required=False, help="List of training run IDs, e.g., run1,run2 or [run1,run2].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, default=None, required=False, help="List of validation run IDs, e.g., run3,run4 or [run3,run4].")
    
    # Parse arguments
    args = parser.parse_args()

    # Parse the CoPick configuration paths
    if len(args.config) > 1:    copick_configs = utils.parse_copick_configs(args.config)
    else:                       copick_configs = args.config[0]    

    # Save JSON with Parameters
    save_parameters_json(args, 'model_explore.json')

    # Call the function with parsed arguments
    model_search(
        copick_config=copick_configs,
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,
        tomo_algorithm=args.tomo_algorithm,
        voxel_size=args.voxel_size,
        Nclass=args.Nclass,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        random_seed=args.random_seed,
        num_epochs=args.num_epochs,
        num_trials=args.num_trials,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs, 
        tomo_batch_size=args.tomo_batch_size,
        best_metric=args.best_metric
    )

def save_parameters_json(args, output_path: str):
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
            "target_name": args.target_name,
            "target_user_id": args.target_user_id,
            "target_session_id": args.target_session_id,
            "tomo_algorithm": args.tomo_algorithm,
            "voxel_size": args.voxel_size,
            "Nclass": args.Nclass,            
        },
        "optimization": {
            "mlflow_experiment_name": args.mlflow_experiment_name,
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
            "random_seed": args.random_seed,
            "num_trials": args.num_trials,
            "best_metric": args.best_metric
        },
        "training": {
            "num_epochs": args.num_epochs,            
            "tomo_batch_size": args.tomo_batch_size,
            "trainRunIDs": args.trainRunIDs,
            "validateRunIDs": args.validateRunIDs,
        }
    }

    # Print the parameters
    print("Parameters for Model ArchitectureSearch:")
    pprint.pprint(params)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)    

if __name__ == "__main__":
    cli()