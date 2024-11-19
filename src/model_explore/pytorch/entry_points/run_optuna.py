from model_explore.pytorch import io, data, hyper_search, utils
import torch, mlflow, os, copick, optuna, argparse
from typing import List

copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"
experimental_mlflow_name = 'fake-search'
segmentation_name = 'segmentation'
num_epochs = 25
num_trials = 10


def model_search(
    copick_config: str,
    mlflow_experiment_name: str = 'model-search',
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000", 
    random_seed: int = 42,
    num_epochs: int = 100,
    num_trials: int = 10,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,   
    ):

    # Split Experiment into Train and Validation Runs
    Nclass = io.get_num_classes(copick_config)
    data_generator = data.train_generator(copick_config, 
                                          segmentation_name, 
                                          Nclasses = Nclass,
                                          tomo_batch_size = 20)
    
    data_generator.get_data_splits(trainRunIDs = trainRunIDs,
                                   validateRunIDs = validateRunIDs)

    # Determine Device to Run Optuna On
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pruning = True
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    )

    # Explicitly initialize the TPE sampler
    tpe_sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,     # Number of initial random trials before TPE kicks in
        n_ei_candidates=24,      # Number of candidate samples for Expected Improvement
        multivariate=True        # Use multivariate TPE for correlated parameter sampling
    )

    # Set up ML-Flow
    utils.mlflow_setup()

    print(f'Running Architecture Search Over 1 GPU\n')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    storage = "sqlite:///example.db"
    with mlflow.start_run():

        study = optuna.create_study(storage=storage,
                                    direction="maximize",
                                    load_if_exists=True,
                                    pruner=pruner)

        mlflow.log_params({"random_seed": random_seed})
        mlflow.log_params(data_generator.get_dataloader_parameters())

        study.optimize(lambda trial: hyper_search.objective(trial, num_epochs, device, data_generator, random_seed = random_seed), 
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

    # Optional arguments
    parser.add_argument("--mlflow-experiment-name", type=str, default="model-search", required=False, help="Name of the MLflow experiment (default: 'model-search').")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://mlflow.mlflow.svc.cluster.local:5000", required=False, help="URI for the MLflow tracking server (default: 'http://mlflow.mlflow.svc.cluster.local:5000').")
    parser.add_argument("--random-seed", type=int, default=42, required=False, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num-epochs", type=int, default=100, required=False, help="Number of epochs per trial (default: 100).")
    parser.add_argument("--num-trials", type=int, default=10, required=False, help="Number of trials for architecture search (default: 10).")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, default=None, required=False, help="List of training run IDs, e.g., run1,run2 or [run1,run2].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, default=None, required=False, help="List of validation run IDs, e.g., run3,run4 or [run3,run4].")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    model_search(
        copick_config=args.config,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        random_seed=args.random_seed,
        num_epochs=args.num_epochs,
        num_trials=args.num_trials,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
    )

if __name__ == "__main__":
    cli()