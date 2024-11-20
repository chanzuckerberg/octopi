from model_explore.pytorch import io, hyper_search, utils
from model_explore.pytorch.datasets import generators
import torch, mlflow, os, copick, optuna, argparse
from typing import List

def model_search(
    copick_config: str,
    target_name: str,
    target_user_id: str = None,
    target_session_id: str = None,
    tomo_algorithm: str = 'wbp',
    voxel_size: float = 10,    
    Nclass: int = 3,
    mlflow_experiment_name: str = 'model-search',
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000", 
    random_seed: int = 42,
    num_epochs: int = 100,
    num_trials: int = 20,
    tomo_batch_size: int = 20,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,   
    ):

    # Split Experiment into Train and Validation Runs
    # Nclass = io.get_num_classes(copick_config)
    data_generator = generators.TrainLoaderManager(copick_config, 
                                                   target_name, 
                                                   target_session_id = target_session_id,
                                                   target_user_id = target_user_id,
                                                   tomo_algorithm = tomo_algorithm,
                                                   voxel_size = voxel_size,                                                    
                                                   Nclasses = Nclass,
                                                   tomo_batch_size = tomo_batch_size)
    
    data_generator.get_data_splits(trainRunIDs = trainRunIDs,
                                   validateRunIDs = validateRunIDs)

    # Get the reload frequency
    data_generator.get_reload_frequency(num_epochs) 

    # Determine Device to Run Optuna On
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pruning = True
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    )

    # Option 1: TPE sampler
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

    # Set up ML-Flow - Start up with defaults for 
    try:
        utils.mlflow_setup()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except:
        pass
    mlflow.set_experiment(mlflow_experiment_name)    

    print(f'Running Architecture Search Over 1 GPU\n')
    storage = "sqlite:///trials.db"
    with mlflow.start_run():

        study = optuna.create_study(storage=storage,
                                    direction="maximize",
                                    sampler=tpe_sampler,
                                    load_if_exists=True,
                                    pruner=pruner)

        mlflow.log_params({"random_seed": random_seed})
        mlflow.log_params(data_generator.get_dataloader_parameters())

        study.optimize(lambda trial: hyper_search.objective(trial, num_epochs, device, 
                                                            data_generator, random_seed = random_seed), 
                        n_trials=num_trials)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")

# Entry point with argparse
def cli():
    """
    CLI entry point for running optuna model archetecture search.
    """
    parser = argparse.ArgumentParser(description="Perform model architecture search with Optuna and MLflow integration.")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
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
    parser.add_argument("--trainRunIDs", type=utils.parse_list, default=None, required=False, help="List of training run IDs, e.g., run1,run2 or [run1,run2].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, default=None, required=False, help="List of validation run IDs, e.g., run3,run4 or [run3,run4].")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    model_search(
        copick_config=args.config,
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
    )

if __name__ == "__main__":
    cli()