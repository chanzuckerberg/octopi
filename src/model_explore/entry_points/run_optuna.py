from model_explore.datasets import generators, multi_config_generator
from model_explore.pytorch import hyper_search
from model_explore.entry_points import common
from model_explore import utils
import torch, mlflow, optuna, argparse, json, os, pprint
import pandas as pd
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
    random_seed: int,
    num_epochs: int,
    num_trials: int,
    tomo_batch_size: int,
    best_metric: str,
    val_interval: int,
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
        tracking_uri = utils.mlflow_setup()
        mlflow.set_tracking_uri(tracking_uri)
    except:
        pass
    mlflow.set_experiment(mlflow_experiment_name)    
    storage = "sqlite:///trials.db"

    # Get available GPUs (for example, on an 4 GPU node) - Run the appropriate function based on the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f'Running Architecture Search Over {gpu_count} GPUs\n')

    if gpu_count > 1:
        multi_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, 
                         random_seed, num_trials, best_metric, val_interval, gpu_count)
    else:
        single_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, 
                          random_seed, num_trials, best_metric, val_interval)
    

def single_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, random_seed, num_trials, best_metric, val_interval):
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
                                                            val_interval = val_interval,
                                                            best_metric = best_metric), 
                        n_trials=num_trials)

        # Save contour plot
        save_contour_plot_as_png(study)

        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_params}")
    
def multi_gpu_optuna(storage, tpe_sampler, pruner, data_generator, num_epochs, random_seed, num_trials, best_metric, val_interval, gpu_count):
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
                                                                      val_interval = val_interval,
                                                                      gpu_count = gpu_count), 
                       n_trials=num_trials,
                       n_jobs=gpu_count) # Run trials on multiple GPUs
        
        # Save contour plot
        save_contour_plot_as_png(study)

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

def optuna_parser(parser_description, add_slurm: bool = False):
    
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input Arguments
    input_group = parser.add_argument_group("Input Arguments")
    common.add_config(input_group, single_config=False)
    input_group.add_argument("--tomo-algorithm", default='wbp', help="Tomogram algorithm used for training")
    input_group.add_argument("--mlflow-experiment-name", type=str, default="model-search", required=False, help="Name of the MLflow experiment (default: 'model-search').")
    input_group.add_argument("--random-seed", type=int, default=42, required=False, help="Random seed for reproducibility (default: 42).")
    input_group.add_argument("--best-metric", type=str, default='avg_f1', required=False, help="Metric to Monitor for Optimization")
    input_group.add_argument("--trainRunIDs", type=utils.parse_list, default=None, required=False, help="List of training run IDs, e.g., run1,run2 or [run1,run2].")
    input_group.add_argument("--validateRunIDs", type=utils.parse_list, default=None, required=False, help="List of validation run IDs, e.g., run3,run4 or [run3,run4].")    

    model_group = parser.add_argument_group("Model Arguments")
    common.add_model_parameters(model_group, model_explore = True)

    train_group = parser.add_argument_group("Training Arguments")
    common.add_train_parameters(train_group, model_explore = True)

    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'optuna')

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    """
    CLI entry point for running optuna model archetecture search.
    """

    description="Perform model architecture search with Optuna and MLflow integration."
    args = optuna_parser(description)

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
        random_seed=args.random_seed,
        num_epochs=args.num_epochs,
        num_trials=args.num_trials,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs, 
        tomo_batch_size=args.tomo_batch_size,
        best_metric=args.best_metric,
        val_interval=args.val_interval
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
    print(f"\nParameters for Model Architecture Search:")
    pprint.pprint(params); print()

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)   

def save_contour_plot_as_png(study):
    """
    Save the contour plot of hyperparameter interactions as a PNG, 
    automatically extracting parameter names from the study object.

    Args:
        study: The Optuna study object.
        output_path: Path to save the PNG file.
    """
    # Extract all parameter names from the study trials
    all_params = set()
    for trial in study.trials:
        all_params.update(trial.params.keys())
    all_params = list(all_params)  # Convert to a sorted list for consistency

    # Generate the contour plot
    fig = optuna.visualization.plot_contour(study, params=all_params) 

    # Adjust figure size and font size
    fig.update_layout(
        width=6000, height=6000,  # Large figure size
        font=dict(size=40)  # Increase font size for better readability
    )

    # Save the plot as a PNG file
    fig.write_image('model_exploration/contour_plot.png', scale=1)  

     # Extract trial data
    trials = [
        {**trial.params, 'objective_value': trial.value}
        for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    # Convert to DataFrame
    df = pd.DataFrame(trials)

    # Save to CSV
    df.to_csv("model_exploration/optuna_results.csv", index=False)


if __name__ == "__main__":
    cli()