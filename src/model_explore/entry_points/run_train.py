from model_explore.datasets import generators, multi_config_generator
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.models import common as builder
from model_explore.entry_points import common 
from model_explore.pytorch import trainer 
from model_explore import io, utils
from monai.metrics import ConfusionMatrixMetric
import torch, mlflow, os, argparse, json
from typing import List, Optional

def train_model(
    copick_config_path: str,
    target_name: str,
    target_user_id: str = None,
    target_session_id: str = None,
    tomo_algorithm: str = 'wbp',
    voxel_size: float = 10,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,    
    channels: List[int] = [32,64,128,128],
    strides: List[int] = [2,2,1],
    res_units: int = 2,
    Nclass: int = 3,
    model_type: str = 'UNet',
    model_save_path: str = 'results',
    model_weights: Optional[str] = None,
    dim_in: int = 96,
    num_tomo_crops: int = 16,
    tomo_batch_size: int = 15,
    lr: float = 1e-3,
    tversky_alpha: float = 0.5,
    num_epochs: int = 100,
    val_interval: int = 25,
    mlflow: bool = False,
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000", 
    mlflow_experiment_name: str = "model-train"    
    ):

    # Initialize the data generator to manage training and validation datasets
    print(f'Training with {copick_config_path}\n')
    if isinstance(copick_config_path, dict):
        # Multi-config training
        data_generator = multi_config_generator.MultiConfigTrainLoaderManager(
            copick_config_path, 
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
            copick_config_path, 
            target_name, 
            target_session_id = target_session_id,
            target_user_id = target_user_id,
            tomo_algorithm = tomo_algorithm,
            voxel_size = voxel_size,
            Nclasses = Nclass,
            tomo_batch_size = tomo_batch_size ) 
    
    # Get the data splits
    data_generator.get_data_splits(trainRunIDs = trainRunIDs,
                                   validateRunIDs = validateRunIDs)
    
    # Get the reload frequency
    data_generator.get_reload_frequency(num_epochs)

    # Monai Functions
    alpha = tversky_alpha
    beta = 1 - alpha
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
    metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")

    # Create UNet Model and Load Weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    channels = [int(channel) for channel in channels.split(',')]
    strides = [int(stride) for stride in strides.split(',')]
    model_builder = builder.get_model(Nclass, device, model_type)
    if model_type == "Unet":            model_builder.build_model(channels, strides, res_units)
    elif model_type == "AttentionUnet": model_builder.build_model(channels, strides)
    model = model_builder.model.to(device)
    if model_weights: 
        model.load_state_dict(torch.load(model_weights, weights_only=True))     

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)

    # Create UNet-Trainer
    model_trainer = trainer.ModelTrainer(model, device, loss_function, metrics_function, optimizer)

    if mlflow:
        run_training_with_mlflow(model_trainer, model, data_generator, model_save_path, 
                                 mlflow_tracking_uri, mlflow_experiment_name, num_epochs, 
                                 num_tomo_crops, val_interval)
    else:
        run_training_local(model_trainer, model, data_generator, model_save_path, num_epochs, 
                           num_tomo_crops, val_interval, dim_in)

def run_training_local(model_trainer, model, data_generator, model_save_path, num_epochs, num_tomo_crops, val_interval, dim_in):

    results = model_trainer.train(
        data_generator, model_save_path, max_epochs=num_epochs,
        crop_size=dim_in, my_num_samples=num_tomo_crops,
        val_interval=val_interval, verbose=True
    )

    # Save parameters and results
    parameters_save_name = os.path.join(model_save_path, "training_parameters.json")
    io.save_parameters_to_json(model, model_trainer, data_generator, parameters_save_name)

    results_save_name = os.path.join(model_save_path, "results.json")
    io.save_results_to_json(results, results_save_name)

def run_training_with_mlflow(model_trainer, model, data_generator, model_save_path, mlflow_tracking_uri, mlflow_experiment_name, num_epochs, num_tomo_crops, val_interval):
    
    try:
        utils.mlflow_setup()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except: pass
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():
        model_trainer.train(
            data_generator, max_epochs=num_epochs,
            my_num_samples=num_tomo_crops, val_interval=val_interval,
            model_save_path=model_save_path, use_mlflow = True, verbose=False
        )

        # Log parameters and metrics
        params = {
            "model": io.get_model_parameters(model),
            "optimizer": io.get_optimizer_parameters(model_trainer),
            "dataloader": data_generator.get_dataloader_parameters()
        }
        mlflow.log_params(io.flatten_params(params))
    mlflow.end_run()

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
    input_group.add_argument("--tomo-algorithm", default='wbp', help="Tomogram algorithm used for training")
    input_group.add_argument("--trainRunIDs", type=utils.parse_list, help="List of training run IDs, e.g., run1,run2,run3")
    input_group.add_argument("--validateRunIDs", type=utils.parse_list, help="List of validation run IDs, e.g., run4,run5,run6")
    input_group.add_argument("--mlflow", type=utils.string2bool, default=False, help="Log the results with MLFlow or save in a unique directory")    
    
    # Model Arguments
    model_group = parser.add_argument_group("Model Arguments")
    common.add_model_parameters(model_group)
    
    # Training Arguments
    train_group = parser.add_argument_group("Training Arguments")
    common.add_train_parameters(train_group)
    
    # SLURM Arguments
    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'train')

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    """
    CLI entry point for training models where results can either be saved to a local directory or a server with MLFlow.
    """

    # Parse the arguments
    parser_description = "Train 3D CNN models"
    args = train_model_parser(parser_description)

    # Parse the CoPick configuration paths
    if len(args.config) > 1:    copick_configs = utils.parse_copick_configs(args.config)
    else:                       copick_configs = args.config[0]

    # Save JSON with Parameters
    # save_parameters_json(args, 'train.json')

    # Call the training function
    train_model(
        copick_config_path=copick_configs, 
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,        
        tomo_algorithm=args.tomo_algorithm,
        voxel_size=args.voxel_size,
        channels=args.channels,
        strides=args.strides,
        res_units=args.res_units,
        Nclass=args.Nclass,
        model_type=args.model_type,
        model_save_path=args.model_save_path,
        dim_in=args.dim_in,
        num_tomo_crops=args.num_tomo_crops,
        tomo_batch_size=args.tomo_batch_size,
        lr=args.lr,
        tversky_alpha=args.tversky_alpha,
        num_epochs=args.num_epochs,
        val_interval=args.val_interval,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,  
        mlflow=args.mlflow,      
    )

# def save_parameters_json(args, output_path: str):    
#     """
#     Save the training parameters to a JSON file.
#     Args:
#         args: Parsed arguments from argparse.
#         output_path: Path to save the JSON file.
#     """
#     # Organize parameters into categories
#     params = {
#         "input": {
#             "copick_config_path": args.config,
#             "target_name": args.target_name,
#             "target_user_id": args.target_user_id,
#             "target_session_id": args.target_session_id,
#             "tomo_algorithm": args.tomo_algorithm,
#             "voxel_size": args.voxel_size,            
#         },
#         "model": {
#             "model_type": args.model_type,
#             "Nclass": args.Nclass,
#             "dim_in": args.dim_in,
#             "channels": args.channels,
#             "strides": args.strides,
#             "res_units": args.res_units,
#         },
#         "training": {
#             "num_tomo_crops": args.num_tomo_crops,
#             "tomo_batch_size": args.tomo_batch_size,
#             "lr": args.lr,
#             "tversky_alpha": args.tversky_alpha,
#             "num_epochs": args.num_epochs,
#             "val_interval": args.val_interval,
#             "trainRunIDs": args.trainRunIDs,
#             "validateRunIDs": args.validateRunIDs,
#         },
#         "output": {
#             "model_save_path": args.model_save_path,
#             "mlflow": args.mlflow,
#         },
#     }

#     # Save to JSON file
#     with open(output_path, 'w') as f:
#         json.dump(params, f, indent=4)    

if __name__ == "__main__":
    cli()