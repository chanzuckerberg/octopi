from model_explore.pytorch.datasets import generators, multi_config_generator
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch import io, trainer, utils
from monai.metrics import ConfusionMatrixMetric
import torch, mlflow, os, argparse, json
from typing import List, Optional

def train_model(
    copick_config_path: str,
    target_name: str,
    target_user_id: str = None,
    target_session_id: str = None,
    tomo_algorithms: List[str] = ['wbp'],
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

    if len(tomo_algorithms) == 1:
        tomo_algorithm = tomo_algorithms[0]

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
    model = utils.create_model(model_type, Nclass, channels, strides, res_units, device)
    if model_weights: 
        model.load_state_dict(torch.load(model_weights, weights_only=True))

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.1)
    # optimizer = torch.optim.RAdam(model.parameters(), lr, weight_decay=0.1)

    # Create UNet-Trainer
    model_trainer = trainer.unet(model, device, loss_function, metrics_function, optimizer)

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

# Entry point with argparse
def cli():
    """
    CLI entry point for training models where results can either be saved to a local directory or a server with MLFlow.
    """

    parser = argparse.ArgumentParser(description="Train a UNet model on CryoET Tomograms.")
    # parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--config", type=str, required=True, action='append',
                            help="Specify a single configuration path (/path/to/config.json) "
                                 "or multiple entries in the format session_name,/path/to/config.json. "
                                 "Use multiple --config entries for multiple sessions.")    
    parser.add_argument("--target-name", type=str, required=True, help="Name of the target segmentation.")
    parser.add_argument("--target-user-id", type=str, required=False, default=None, help="User ID for the target segmentation.")
    parser.add_argument("--target-session-id", type=str, required=False, default=None, help="Session ID for the target segmentation.")
    parser.add_argument("--tomo-algorithms", type=utils.parse_list, required=False, default = 'wbp', help="Tomogram algorithms to use for training.")
    parser.add_argument("--voxel-size", type=float, required=False, default = 10, help="Voxel size for the tomograms.")
    parser.add_argument("--channels", type=utils.parse_int_list, required=False, default = [32,64,128,128], help="List of channel sizes for each layer, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, required=False, default = [2,2,1], help="List of stride sizes for each layer, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res-units", type=int, required=False, default = 2, help="Number of residual units in the UNet.")
    parser.add_argument("--dim-in", type=int, required=False, default = 96, help="Input dimension for the UNet model.")
    parser.add_argument("--Nclass", type=int, required=False, default = 3, help="Number of prediction classes in the model.")
    parser.add_argument("--model-type", type=str, required=False, default = 'UNet', help="Type of model to use. Available options: ['UNet', 'AttentionUNet']")
    parser.add_argument("--model-save-path", type=str, required=False, default='results', help="Path to save the trained model and results.")
    parser.add_argument("--num-tomo-crops", type=int, required=False, default = 16, help="Number of tomogram crops to use.")
    parser.add_argument("--tomo-batch-size", type=int, required=False, default=15, help="Number of tomograms to load per epoch for training.")
    parser.add_argument("--lr", type=float, required=False, default = 1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--tversky-alpha", type=float, required=False, default = 0.5, help="Alpha parameter for the Tversky loss.")
    parser.add_argument("--num-epochs", type=int, required=False, default = 100, help="Number of training epochs.")
    parser.add_argument("--val-interval", type=int, required=False, default=25, help="Interval for validation metric calculations.")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, required=False, help="List of training run IDs, e.g., run1,run2,run3 or [run1,run2,run3].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, required=False, help="List of validation run IDs, e.g., run4,run5,run6 or [run4,run5,run6].")    
    parser.add_argument("--mlflow", type=utils.string2bool, required=False, default=False, help="Log the Results with MLFlow or Save it A Unique Directory.")

    args = parser.parse_args()

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
        tomo_algorithms=args.tomo_algorithms,
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