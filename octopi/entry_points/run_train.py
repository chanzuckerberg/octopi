from typing import List, Optional, Tuple
from octopi.entry_points import common
from octopi.utils import parsers
from octopi import cli_context
import rich_click as click

def train_model(
    copick_config_path: str,
    target_info: Tuple[str, str, str],
    tomo_algorithm: str = 'wbp',
    voxel_size: float = 10,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,    
    model_config: str = None,
    model_weights: Optional[str] = None,
    output: str = 'results',
    batch_size: int = 16,
    ncache_tomos: int = 15,
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

    from octopi.datasets import generators
    from monai.losses import TverskyLoss
    from octopi.utils import parsers, io
    from octopi.workflows import train

    # # Multi-config training
    if isinstance(copick_config_path, dict):
        data_generator = generators.MultiCopickDataModule(
            copick_config_path, 
            tomo_algorithm,
            target_info[0], 
            sessionid = target_info[2],
            userid = target_info[1],
            voxel_size = voxel_size,
            tomo_batch_size = ncache_tomos )
    else:  # Single-config training
        data_generator = generators.CopickDataModule(
            copick_config_path, 
            tomo_algorithm,
            target_info[0], 
            sessionid = target_info[2],
            userid = target_info[1],
            voxel_size = voxel_size,
            tomo_batch_size = ncache_tomos )

    # Get the data splits and Nclasses
    ratios = parsers.parse_data_split(data_split)
    data_generator.get_data_splits(
        trainRunIDs = trainRunIDs,
        validateRunIDs = validateRunIDs,
        train_ratio = ratios[0], val_ratio = ratios[1], test_ratio = ratios[2],
        create_test_dataset = False)
    model_config['num_classes'] = data_generator.Nclasses

    # Loss Functions
    alpha = tversky_alpha
    beta = 1 - alpha
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
    
    # Train the Model
    train(
        data_generator, loss_function, 
        model_config = model_config, model_weights = model_weights,
        best_metric = best_metric, num_epochs = num_epochs,
        model_save_path = output, lr0 = lr,
        batch_size = batch_size,
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

@click.command('train')
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
              help="Tomogram algorithm used for training, provide a comma-separated list of algorithms for multiple options. (e.g., 'denoised,wbp')")
@click.option('-tinfo', "--target-info", type=str, default="targets,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target information, e.g., 'name' or 'name,user_id,session_id'. Default is 'targets,octopi,1'.")
@common.config_parameters(single_config=False)
def cli(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, ncache_tomos, best_metric, 
        batch_size, lr, tversky_alpha, output):
    """
    Train 3D CNN U-Net models for Cryo-ET semantic segmentation.
    """

    print('\n🚀 Training a New Octopi Model...\n')
    run_train(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, ncache_tomos, best_metric, 
        batch_size, lr, tversky_alpha, output)

def run_train(config, voxel_size, target_info, tomo_alg, trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, ncache_tomos, best_metric, 
        batch_size, lr, tversky_alpha, output):
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
        output=output,
        batch_size=batch_size,
        ncache_tomos=ncache_tomos,
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