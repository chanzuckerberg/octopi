from typing import List, Optional, Tuple
from octopi.entry_points import common
from octopi.utils import parsers
from octopi import cli_context
import rich_click as click

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

def train_model(
    copick_config_path: str,
    target_info: Tuple[str, str, str],
    tomo_algorithm: str = 'wbp',
    voxel_size: float = 10,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,    
    model_config: str = None,
    model_weights: Optional[str] = None,
    model_save_path: str = 'results',
    num_tomo_crops: int = 16,
    tomo_batch_size: int = 15,
    lr: float = 1e-3,
    tversky_alpha: float = 0.5,
    num_epochs: int = 100,  
    val_interval: int = 5,
    best_metric: str = 'avg_f1',
    data_split: str = '0.8'
    ):
    """
    Train a 3D U-Net model using the specified CoPick configuration and target information.
    """

    import matplotlib
    # Force a headless-safe backend everywhere (must be BEFORE pyplot import)
    matplotlib.use("Agg", force=True)

    from octopi.datasets import generators, multi_config_generator
    from monai.losses import TverskyLoss
    from octopi.utils import parsers, io
    from octopi.workflows import train

    # Initialize the data generator to manage training and validation datasets
    print(f'Training with {copick_config_path}\n')

    # Multi-config training
    if isinstance(copick_config_path, dict):
        data_generator = multi_config_generator.MultiConfigTrainLoaderManager(
            copick_config_path, 
            target_info[0], 
            target_session_id = target_info[2],
            target_user_id = target_info[1],
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            tomo_batch_size = tomo_batch_size )
    else:  # Single-config training
        data_generator = generators.TrainLoaderManager(
            copick_config_path, 
            target_info[0], 
            target_session_id = target_info[2],
            target_user_id = target_info[1],
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            tomo_batch_size = tomo_batch_size )

    # Get the data splits
    ratios = parsers.parse_data_split(data_split)
    data_generator.get_data_splits(
        trainRunIDs = trainRunIDs,
        validateRunIDs = validateRunIDs,
        train_ratio = ratios[0], val_ratio = ratios[1], test_ratio = ratios[2],
        create_test_dataset = False)
    
    # Get the reload frequency
    data_generator.get_reload_frequency(num_epochs)
    model_config['num_classes'] = data_generator.Nclasses

    # Monai Functions
    alpha = tversky_alpha
    beta = 1 - alpha
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
    
    # Train the Model
    train(
        data_generator, loss_function, 
        model_config = model_config, model_weights = model_weights,
        best_metric = best_metric, num_epochs = num_epochs,
        model_save_path = model_save_path, lr0 = lr
    )

def get_model_config(channels, strides, res_units, dim_in):
    """
    Create a model configuration dictionary if no model configuration file is provided.
    """
    model_config = {
        'architecture': 'Unet',
        'channels': channels,
        'strides': strides,
        'num_res_units': res_units, 
        'dropout': 0.1,
        'dim_in': dim_in
    }
    return model_config

@click.command('train', help="Train 3D CNN U-Net models")
# Training Arguments (applied in reverse order)
@common.train_parameters(octopi=False)
# UNet-Model Arguments
@common.model_parameters(octopi=False)
# Fine-Tuning Arguments
@click.option('-mw', '--model-weights', type=click.Path(exists=True), default=None,
              help="Path to the model weights file (typically used for fine-tuning)")
@click.option('-mc', '--model-config', type=click.Path(exists=True), default=None,
              help="Path to the model configuration file (typically used for fine-tuning)")
# Input Arguments
@click.option('-split', '--data-split', type=str, default='0.8',
              help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")
@click.option('-vruns', "--validateRunIDs", type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of validation run IDs, e.g., run4,run5,run6")
@click.option('-truns', "--trainRunIDs", type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of training run IDs, e.g., run1,run2,run3")
@click.option('-alg',"--tomo-alg", type=str, default='wbp',
              help="Tomogram algorithm used for training")
@click.option('-tinfo', "--target-info", type=str, default="targets,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target information, e.g., 'name' or 'name,user_id,session_id'. Default is 'targets,octopi,1'.")
@common.config_parameters(single_config=False)
def cli(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, tomo_batch_size, best_metric, 
        num_tomo_crops, lr, tversky_alpha, model_save_path):
    """
    CLI entry point for training models where results can either be saved to a local directory or a server with MLFlow.
    """

    run_train(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, tomo_batch_size, best_metric, 
        num_tomo_crops, lr, tversky_alpha, model_save_path)

def run_train(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, tomo_batch_size, best_metric, 
        num_tomo_crops, lr, tversky_alpha, model_save_path):
    """
    Run the training model.
    """
    import octopi.utils.io as io

    # Parse the CoPick configuration paths
    if len(config) > 1:
        copick_configs = parsers.parse_copick_configs(config)
    else:
        copick_configs = config[0]
    
    if model_config:
        model_config_dict = io.load_yaml(model_config)
    else:
        model_config_dict = get_model_config(channels, strides, res_units, dim_in)

    # Call the training function
    train_model(
        copick_config_path=copick_configs, 
        target_info=target_info,
        tomo_algorithm=tomo_alg,
        voxel_size=voxel_size,
        model_config=model_config_dict,
        model_weights=model_weights,
        model_save_path=model_save_path,
        num_tomo_crops=num_tomo_crops,
        tomo_batch_size=tomo_batch_size,
        lr=lr,
        tversky_alpha=tversky_alpha,
        num_epochs=num_epochs,
        val_interval=val_interval,
        best_metric=best_metric,
        trainRunIDs=trainrunids,
        validateRunIDs=validaterunids,
        data_split=data_split
    )

if __name__ == '__main__':
    cli()