from octopi.entry_points import common
from octopi.utils import parsers
import rich_click as click

def save_parameters(config: tuple,
                    target_info: tuple,
                    tomo_alg: str,
                    voxel_size: float,
                    model_type: str,
                    mlflow_experiment_name: str,
                    random_seed: int,
                    num_trials: int,
                    best_metric: str,
                    num_epochs: int,
                    tomo_batch_size: int,
                    trainRunIDs: list,
                    validateRunIDs: list,
                    data_split: str,
                    output_path: str):
    """
    Save the Optuna search parameters to a YAML file.
    """
    import octopi.utils.io as io
    import pprint

    # Organize parameters into categories
    params = {
        "input": {
            "copick_config": config,
            "target_info": target_info,
            "tomo_algorithm": tomo_alg,
            "voxel_size": voxel_size,          
        },
        "optimization": {
            "model_type": model_type,
            "mlflow_experiment_name": mlflow_experiment_name,
            "random_seed": random_seed,
            "num_trials": num_trials,
            "best_metric": best_metric
        },
        "training": {
            "num_epochs": num_epochs,            
            "tomo_batch_size": tomo_batch_size,
            "trainRunIDs": trainRunIDs,
            "validateRunIDs": validateRunIDs,
            "data_split": data_split
        }
    }

    # Print the parameters
    print(f"\nParameters for Model Architecture Search:")
    pprint.pprint(params); print()

    # Save to YAML file
    io.save_parameters_yaml(params, output_path)


@click.command('model-explore', help="Perform model architecture search with Optuna and MLflow integration")
# Training Arguments
@click.option('--random-seed', type=int, default=42,
              help="Random seed for reproducibility")
@common.train_parameters(octopi=True)
# Model Arguments
@click.option('--model-type', type=click.Choice(['Unet', 'AttentionUnet', 'MedNeXt', 'SegResNet'], case_sensitive=False),
              default='Unet',
              help="Model type to use for training")
# Input Arguments
@click.option('-split', '--data-split', type=str, default='0.8',
              help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")
@click.option('-vruns', '--validateRunIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of validation run IDs, e.g., run3,run4 or [run3,run4]")
@click.option('-truns', '--trainRunIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of training run IDs, e.g., run1,run2 or [run1,run2]")
@click.option('--mlflow-experiment-name', type=str, default="model-search",
              help="Name of the MLflow experiment")
@click.option('-alg', '--tomo-alg', type=str, default='wbp',
              help="Tomogram algorithm used for training")
@click.option('-tinfo', '--target-info', type=str, default="targets,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target information, e.g., 'name' or 'name,user_id,session_id'")
@common.config_parameters(single_config=False)
def cli(config, voxel_size, target_info, tomo_alg, mlflow_experiment_name, 
        trainRunIDs, validateRunIDs, data_split,
        model_type,
        num_epochs, val_interval, tomo_batch_size, best_metric, num_trials, random_seed):
    """
    CLI entry point for running optuna model architecture search.
    """

def run_model_explore(config, voxel_size, target_info, tomo_alg, mlflow_experiment_name, 
        trainRunIDs, validateRunIDs, data_split,
        model_type,
        num_epochs, val_interval, tomo_batch_size, best_metric, num_trials, random_seed):
    """
    Run the model exploration.
    """
    from octopi.pytorch.model_search_submitter import ModelSearchSubmit
    import os

    # Parse the CoPick configuration paths
    if len(config) > 1:
        copick_configs = parsers.parse_copick_configs(config)
    else:
        copick_configs = config[0]

    # Create the model exploration directory
    os.makedirs(f'explore_results_{model_type}', exist_ok=True)

    # Save parameters
    save_parameters(
        config=config,
        target_info=target_info,
        tomo_alg=tomo_alg,
        voxel_size=voxel_size,
        model_type=model_type,
        mlflow_experiment_name=mlflow_experiment_name,
        random_seed=random_seed,
        num_trials=num_trials,
        best_metric=best_metric,
        num_epochs=num_epochs,
        tomo_batch_size=tomo_batch_size,
        trainRunIDs=trainRunIDs,
        validateRunIDs=validateRunIDs,
        data_split=data_split,
        output_path=f'explore_results_{model_type}/octopi.yaml'
    )

    # Call the function with parsed arguments
    search = ModelSearchSubmit(
        copick_config=copick_configs,
        target_name=target_info[0],
        target_user_id=target_info[1],
        target_session_id=target_info[2],
        tomo_algorithm=tomo_alg,
        voxel_size=voxel_size,
        model_type=model_type,
        mlflow_experiment_name=mlflow_experiment_name,
        random_seed=random_seed,
        num_epochs=num_epochs,
        num_trials=num_trials,
        trainRunIDs=trainRunIDs,
        validateRunIDs=validateRunIDs, 
        tomo_batch_size=tomo_batch_size,
        best_metric=best_metric,
        val_interval=val_interval,
        data_split=data_split
    )

    # Run the model search
    search.run_model_search()


if __name__ == "__main__":
    cli()