from octopi.entry_points import common
from octopi.utils import parsers
import rich_click as click

@click.command('model-explore')
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
@click.option('--study-name', type=str, default="model-search",
              help="Name of the Optuna/MLflow experiment")
@click.option('-alg', '--tomo-alg', type=str, default='wbp',
              help="Tomogram algorithm used for training, provide a comma-separated list of algorithms for multiple options. (e.g., 'denoised,wbp')")
@click.option('-tinfo', '--target-info', type=str, default="targets,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target information, e.g., 'name' or 'name,user_id,session_id'")
@click.option('-o', '--output', type=str, default='explore_results',
              help="Name of the output directory")
@common.config_parameters(single_config=False)
def cli(
    config, voxel_size, target_info, tomo_alg, study_name, 
    trainrunids, validaterunids, data_split, model_type, num_epochs, background_ratio,
    val_interval, ncache_tomos, best_metric, num_trials, random_seed, output):
    """
    Perform model architecture search with Optuna.
    """

    print('\n🚀 Starting a new Octopi Model Architecture Search...\n')
    run_model_explore(
        config, voxel_size, target_info, tomo_alg, study_name, 
        trainrunids, validaterunids, data_split, model_type, background_ratio, 
        num_epochs, val_interval, ncache_tomos, best_metric, num_trials, random_seed, output
    )

def run_model_explore(config, voxel_size, target_info, tomo_alg, study_name, 
        trainrunids, validaterunids, data_split, model_type, background_ratio,
        num_epochs, val_interval, ncache_tomos, best_metric, num_trials, random_seed, output):
    """
    Run the model exploration.
    """
    from octopi.pytorch.submit_search import ExploreSubmitter
    import os

    # Parse the CoPick configuration paths
    if len(config) > 1:
        copick_configs = parsers.parse_copick_configs(config)
    else:
        copick_configs = config[0]

    # Create the model exploration directory
    os.makedirs(output, exist_ok=True)

    # Call the function with parsed arguments
    search = ExploreSubmitter(
        copick_config=copick_configs,
        target_name=target_info[0],
        target_user_id=target_info[1],
        target_session_id=target_info[2],
        tomo_algorithm=tomo_alg,
        voxel_size=voxel_size,
        model_type=model_type,
        random_seed=random_seed,
        num_epochs=num_epochs,
        num_trials=num_trials,
        trainRunIDs=trainrunids,
        validateRunIDs=validaterunids, 
        ntomo_cache=ncache_tomos,
        best_metric=best_metric,
        val_interval=val_interval,
        data_split=data_split,
        background_ratio=background_ratio
    )

    # Run the model search
    search.run_model_search(study_name, output)


if __name__ == "__main__":
    cli()