from typing import List, Optional, Tuple, Union
from octopi.entry_points import common
from octopi.utils import parsers
import rich_click as click

def train_model(
    copick_config_path: str,
    target_info: Tuple[str, str, str],
    tomo_uris: Union[List[str], str],
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
    data_split: str = '0.8',
    background_ratio: float = 0.0
    ):
    """
    Train a 3D U-Net model using the specified CoPick configuration and target information.
    """

    import matplotlib
    # Force a headless-safe backend everywhere (must be BEFORE pyplot import)
    matplotlib.use("Agg", force=True)

    from octopi.datasets.config import DataGeneratorConfig
    from monai.losses import TverskyLoss
    from octopi.models import common as model_common
    from octopi.workflows import train

    # Create a data generator
    cfg = DataGeneratorConfig(
        config=copick_config_path,
        name=target_info[0], user_id=target_info[1], session_id=target_info[2],
        tomo_uris=tomo_uris, ntomo_cache=ncache_tomos,
        background_ratio=background_ratio, data_split=data_split,
        trainRunIDs=trainRunIDs, validateRunIDs=validateRunIDs
    )
    data_generator = cfg.create_data_generator()
    model_config['num_classes'] = data_generator.Nclasses

    # Loss Functions.
    # If the model config records a base loss in its `optimizer:` block, rebuild that exact loss
    # (e.g. a FocalLoss/gamma chosen by model-explore). Otherwise default to TverskyLoss(alpha).
    opt = model_config.get('optimizer', {}) if isinstance(model_config, dict) else {}
    cfg_loss = opt.get('loss_function')
    if cfg_loss and cfg_loss != 'TverskyLoss' and cfg_loss != 'DeepSupervisionLoss':
        loss_function = model_common.get_loss_function(
            loss_name=cfg_loss,
            gamma=opt.get('gamma'), alpha=opt.get('alpha'),
            weight_tversky=opt.get('weight_tversky'),
        )
        print(f"Using loss from model config: {cfg_loss} "
              f"(gamma={opt.get('gamma')}, alpha={opt.get('alpha')}, "
              f"weight_tversky={opt.get('weight_tversky')})")
    else:
        if cfg_loss == 'DeepSupervisionLoss':
            print(
                "[Warning] model_config records loss_function: DeepSupervisionLoss -- a legacy "
                "save-path artifact that lost the base loss and its gamma/alpha. The original loss "
                "cannot be rebuilt from this config; falling back to TverskyLoss. Re-run "
                "`octopi model-explore` to regenerate the config if you need the exact loss."
            )
        if cfg_loss == 'TverskyLoss' and opt.get('alpha') is not None:
            alpha = opt['alpha']
        else:
            alpha = tversky_alpha
        beta = 1 - alpha
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)
        print(f"Using TverskyLoss (alpha={alpha})")

    # Read per-class score weights from copick config metadata (score_weight key)
    import copick as _copick
    if isinstance(copick_config_path, str):
        _config_path = copick_config_path
    elif isinstance(copick_config_path, dict):
        # Multi-session training: parse_copick_configs returns {session_name: path}.
        # Use the first config (score-weight metadata is shared across sessions).
        _config_path = next(iter(copick_config_path.values()))
    else:
        _config_path = copick_config_path[0]
    _root = _copick.from_file(_config_path)
    class_weights = {
        obj.name: obj.metadata.get('weight', 1)
        for obj in _root.pickable_objects if obj.is_particle
    }

    # Train the Model
    train(
        data_generator, loss_function,
        model_config = model_config, model_weights = model_weights,
        best_metric = best_metric, num_epochs = num_epochs,
        model_save_path = output, lr0 = lr, val_interval = val_interval,
        batch_size = batch_size, class_weights = class_weights,
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

@click.command('train', no_args_is_help=True)
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
@click.option('-turi', "--target-uri", type=str, default="targets:octopi/1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target query as 'name', 'name:user_id', or 'name:user_id/session_id'. Default 'targets:octopi/1'.")
@common.config_parameters(single_config=False)
def cli(
    config, tomo_uris, target_uri, trainrunids, validaterunids, data_split,
    model_config, model_weights,
    channels, strides, res_units, dim_in,
    num_epochs, val_interval, ncache_tomos, best_metric,
    batch_size, lr, tversky_alpha, background_ratio, output):
    """
    Train 3D CNN U-Net models for Cryo-ET semantic segmentation.
    """

    print('\n🚀 Training a New Octopi Model...\n')
    # click `multiple=True` yields a tuple; normalize to a list for downstream use.
    tomo_uris = list(tomo_uris)
    run_train(config, tomo_uris, target_uri,  trainrunids, validaterunids, data_split,
        model_config, model_weights,
        channels, strides, res_units, dim_in,
        num_epochs, val_interval, ncache_tomos, best_metric, 
        batch_size, lr, tversky_alpha, background_ratio, output)

def run_train(
    config, tomo_uris, target_info, trainrunids, validaterunids, data_split,
    model_config, model_weights,
    channels, strides, res_units, dim_in,
    num_epochs, val_interval, ncache_tomos, best_metric, 
    batch_size, lr, tversky_alpha, background_ratio, output
    ):
    """
    Run the training model.
    """
    import octopi.utils.io as io

    # Parse the CoPick configuration paths
    if len(config) > 1:
        copick_configs = parsers.parse_copick_configs(config)
    else:
        copick_configs = config[0]
    
    # Load the model configuration
    if model_config:
        model_config_dict = io.load_yaml(model_config)
    else:
        model_config_dict = get_model_config(channels, strides, res_units, dim_in)

    # Call the training function
    train_model(
        copick_config_path=copick_configs, 
        target_info=target_info,
        tomo_uris=tomo_uris,
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
        data_split=data_split,
        background_ratio=background_ratio
    )

if __name__ == '__main__':
    cli()