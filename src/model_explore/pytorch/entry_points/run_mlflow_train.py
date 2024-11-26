from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch.datasets import generators
from model_explore.pytorch import io, trainer, utils
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import UNet
import torch, mlflow, argparse
from typing import List

def train_model_on_mflow(
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
    model_save_path: str = None,
    model_weights: str = None, 
    num_tomo_crops: int = 16,
    tomo_batch_size: int = 20,
    lr: float = 1e-3,
    tversky_alpha: float = 0.5,    
    num_epochs: int = 100,
    val_interval: int = 25,
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000", 
    mlflow_experiment_name: str = "model-train"
    ):

    # Split Experiment into Train and Validation Runs
    data_generator = generators.TrainLoaderManager(copick_config_path, 
                                                   target_name, 
                                                   target_session_id = target_session_id,
                                                   target_user_id = target_user_id,
                                                   tomo_algorithm = tomo_algorithm,
                                                   voxel_size = voxel_size,                                                   
                                                   Nclasses = Nclass,
                                                   tomo_batch_size = tomo_batch_size)
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Create UNet-Trainer
    train = trainer.unet(model, device, loss_function, metrics_function, optimizer)

    # Set up ML-Flow
    utils.mlflow_setup()

    # Log ML-Flow Refrence and Start Training
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():

        train.mlflow_train(data_generator, 
                        max_epochs = num_epochs,
                        my_num_samples = num_tomo_crops,
                        val_interval = val_interval,
                        model_save_path = model_save_path,
                        verbose=False )
        
        # Log training parameters.
        params = {
            'model': io.get_model_parameters(model),
            'optimizer': io.get_optimizer_parameters(train),
            'dataloader': data_generator.get_dataloader_parameters()
        }
        mlflow.log_params(io.flatten_params(params))

    mlflow.end_run()

# Entry point with argparse
def cli():

    parser = argparse.ArgumentParser(
        description="Train a UNet model on CryoET Tomograms with MLflow integration."
    )

    # Add arguments
    parser.add_argument("--copick-config-path", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, default=None, help="List of training run IDs, e.g., run1,run2,run3 or [run1,run2,run3].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, default=None, help="List of validation run IDs, e.g., run4,run5,run6 or [run4,run5,run6].")
    parser.add_argument("--channels", type=utils.parse_int_list, default = [32,64,128,128], help="List of channel sizes for the UNet model, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, default = [2,2,1], help="List of stride sizes for the UNet model, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res-units", type=int, required=False, default=2, help="Number of residual units in the UNet model.")
    parser.add_argument("--Nclass", type=int, required=False, default=3, help="Number of prediction classes in the model.")
    parser.add_argument("--model-type", type=str, required=False, default = 'UNet', help="Type of model to use.")
    parser.add_argument("--model-save-path", type=str, required=False, default=None, help="Path to save the trained model.")
    parser.add_argument("--model-weights", type=str, required=False, default=None, help="Path to the pretrained model weights.")
    parser.add_argument("--target-name", type=str, required=True, help="Copick Name of the target segmentation for training.")
    parser.add_argument("--target-user-id", type=str, required=False, default=None, help="User ID of the target segmentation for training.")
    parser.add_argument("--target-session-id", type=str, required=False, default=None, help="Session ID of the target segmentation for training.")
    parser.add_argument("--num-tomo-crops", type=int, required=False, default = 16, help="Number of tomographic crops per training batch.")
    parser.add_argument("--tomo-batch-size", type=int, required=False, default = 25, help="Number of tomograms to load per epoch for training.")
    parser.add_argument("--lr", type=float, required=False, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--tversky-alpha", type=float, required=False, default = 0.5, help="Alpha parameter for the Tversky loss.")
    parser.add_argument("--num-epochs", type=int, required=False, default=100, help="Number of epochs for training.")
    parser.add_argument("--val-interval", type=int, required=False, default=15, help="Number of epochs to wait prior to measuring validation metrics.")    
    parser.add_argument("--mlflow-tracking-uri", type=str, required=False, default = "http://mlflow.mlflow.svc.cluster.local:5000", help="MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment-name", type=str, required=False, default = "model_train", help="Name of the MLflow experiment.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    train_model_on_mflow(
        copick_config_path=args.copick_config_path,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
        channels=args.channels,
        strides=args.strides,
        res_units=args.res_units,
        Nclass=args.Nclass,
        model_type=args.model_type,        
        model_save_path=args.model_save_path,
        model_weights=args.model_weights,
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,
        num_tomo_crops=args.num_tomo_crops,
        tomo_batch_size=args.tomo_batch_size,
        lr=args.lr,
        tversky_alpha=args.tversky_alpha,        
        num_epochs=args.num_epochs,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )

if __name__ == "__main__":
    cli()