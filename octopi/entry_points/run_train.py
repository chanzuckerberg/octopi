from octopi.datasets import generators, multi_config_generator
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from octopi.models import common as builder
from monai.metrics import ConfusionMatrixMetric
from octopi.utils import parsers, io
from octopi.entry_points import common 
from octopi.pytorch import trainer 
import torch, os, argparse
from typing import List, Optional, Tuple

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

    # Initialize the data generator to manage training and validation datasets
    print(f'Training with {copick_config_path}\n')
    if isinstance(copick_config_path, dict):
        # Multi-config training
        data_generator = multi_config_generator.MultiConfigTrainLoaderManager(
            copick_config_path, 
            target_info[0], 
            target_session_id = target_info[2],
            target_user_id = target_info[1],
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = model_config['num_classes'],
            tomo_batch_size = tomo_batch_size )
    else:
        # Single-config training
        data_generator = generators.TrainLoaderManager(
            copick_config_path, 
            target_info[0], 
            target_session_id = target_info[2],
            target_user_id = target_info[1],
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = model_config['num_classes'],
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

    # Monai Functions
    alpha = tversky_alpha
    beta = 1 - alpha
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
    metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")
    
    # Build the Model
    model_builder = builder.get_model(model_config['architecture'])
    model = model_builder.build_model(model_config)
    
    # Load the Model Weights if Provided 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_weights: 
        state_dict = torch.load(model_weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)     
    model.to(device) 

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)

    # Create UNet-Trainer
    model_trainer = trainer.ModelTrainer(model, device, loss_function, metrics_function, optimizer)

    results = model_trainer.train(
        data_generator, model_save_path, max_epochs=num_epochs,
        crop_size=model_config['dim_in'], my_num_samples=num_tomo_crops,
        val_interval=val_interval, best_metric=best_metric, verbose=True
    )
    
    # Save parameters and results
    parameters_save_name = os.path.join(model_save_path, "model_config.yaml")
    io.save_parameters_to_yaml(model_builder, model_trainer, data_generator, parameters_save_name)

    # TODO: Write Results to Zarr or Another File Format? 
    results_save_name = os.path.join(model_save_path, "results.json")
    io.save_results_to_json(results, results_save_name)

def train_model_parser(parser_description, add_slurm: bool = False):
    """
    Parse the arguments for the training model
    """
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input Arguments
    input_group = parser.add_argument_group("Input Arguments")
    common.add_config(input_group, single_config=False)
    input_group.add_argument("--target-info", type=parsers.parse_target, default="targets,octopi,1", 
                             help="Target information, e.g., 'name' or 'name,user_id,session_id'. Default is 'targets,octopi,1'.")
    input_group.add_argument("--tomo-alg", default='wbp', help="Tomogram algorithm used for training")
    input_group.add_argument("--trainRunIDs", type=parsers.parse_list, help="List of training run IDs, e.g., run1,run2,run3")
    input_group.add_argument("--validateRunIDs", type=parsers.parse_list, help="List of validation run IDs, e.g., run4,run5,run6")
    input_group.add_argument('--data-split', type=str, default='0.8', help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) "
                                "or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")
    
    fine_tune_group = parser.add_argument_group("Fine-Tuning Arguments")
    fine_tune_group.add_argument('--model-config', type=str, help="Path to the model configuration file (typically used for fine-tuning)")
    fine_tune_group.add_argument('--model-weights', type=str, help="Path to the model weights file (typically used for fine-tuning)") 

    # Model Arguments
    model_group = parser.add_argument_group("UNet-Model Arguments")
    common.add_model_parameters(model_group)   
    
    # Training Arguments
    train_group = parser.add_argument_group("Training Arguments")
    common.add_train_parameters(train_group)
    
    # SLURM Arguments
    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'train', gpus = 1)

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    """
    CLI entry point for training models where results can either be saved to a local directory or a server with MLFlow.
    """

    # Parse the arguments
    parser_description = "Train 3D CNN U-Net models"
    args = train_model_parser(parser_description)

    # Parse the CoPick configuration paths
    if len(args.config) > 1:    copick_configs = parsers.parse_copick_configs(args.config)
    else:                       copick_configs = args.config[0]

    if args.model_config:
        model_config = io.load_yaml(args.model_config)
    else:
        model_config = get_model_config(args.channels, args.strides, args.res_units, args.Nclass, args.dim_in)

    # Call the training function
    train_model(
        copick_config_path=copick_configs, 
        target_info=args.target_info,
        tomo_algorithm=args.tomo_alg,
        voxel_size=args.voxel_size,
        model_config=model_config,
        model_weights=args.model_weights,
        model_save_path=args.model_save_path,
        num_tomo_crops=args.num_tomo_crops,
        tomo_batch_size=args.tomo_batch_size,
        lr=args.lr,
        tversky_alpha=args.tversky_alpha,
        num_epochs=args.num_epochs,
        val_interval=args.val_interval,
        best_metric=args.best_metric,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
        data_split=args.data_split
    )

def get_model_config(channels, strides, res_units, Nclass, dim_in):
    """
        Create a model configuration dictionary if no model configuration file is provided.
    """
    model_config = {
        'architecture': 'Unet',
        'channels': channels,
        'strides': strides,
        'num_res_units': res_units, 
        'num_classes': Nclass,
        'dropout': 0.1,
        'dim_in': dim_in
    }
    return model_config

if __name__ == "__main__":
    cli()