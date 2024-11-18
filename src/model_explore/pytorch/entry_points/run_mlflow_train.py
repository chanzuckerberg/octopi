from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch import io, trainer, utils, data
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import UNet
import torch, mlflow, argparse
from typing import List

def train_model_on_mflow(
    copick_config_path: str,
    target_name: str,
    target_user_id: str = None,
    target_session_id: str = None,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,    
    channels: List[int] = [32,64,128,128],
    strides: List[int] = [2,2,1],
    res_units: int = 2,
    model_save_path: str = None,
    model_weights: str = None, 
    num_tomo_crops: int = 16,
    reload_frequency: int = 25,
    lr: float = 1e-3,
    num_epochs: int = 100,
    val_interval: int = 25,
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000", 
    mlflow_experiment_name: str = "model-train"
    ):

    # Split Experiment into Train and Validation Runs
    Nclass = io.get_num_classes(copick_config_path)
    data_generator = data.train_generator(copick_config_path, 
                                          target_name, 
                                          target_session_id = target_session_id,
                                          target_user_id = target_user_id,
                                          Nclasses = Nclass,
                                          tomo_batch_size = 20)
    
    data_generator.get_data_splits(trainRunIDs = trainRunIDs,
                                   validateRunIDs = validateRunIDs)

    # Monai Functions
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  
    metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")

    # Create UNet Model and Load Weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=Nclass,
        channels=channels,
        strides=strides,
        num_res_units=res_units,
    ).to(device)
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
                        reload_frequency = reload_frequency,
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
        description="Train a UNet model with MLflow integration."
    )

    # Add arguments
    parser.add_argument("--copick_config_path", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, default=None, help="List of training run IDs, e.g., run1,run2,run3 or [run1,run2,run3].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, default=None, help="List of validation run IDs, e.g., run4,run5,run6 or [run4,run5,run6].")
    parser.add_argument("--channels", type=utils.parse_int_list, default="32,64,128,128", help="List of channel sizes for the UNet model, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, default="2,2,1", help="List of stride sizes for the UNet model, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res_units", type=int, required=False, default=2, help="Number of residual units in the UNet model.")
    parser.add_argument("--model_save_path", type=str, required=False, default=None, help="Path to save the trained model.")
    parser.add_argument("--model_weights", type=str, required=False, default=None, help="Path to the pretrained model weights.")
    parser.add_argument("--target_name", type=str, required=True, help="Copick Name of the target segmentation for training.")
    parser.add_argument("--target_user_id", type=str, required=False, default=None, help="User ID of the target segmentation for training.")
    parser.add_argument("--target_session_id", type=str, required=False, default=None, help="Session ID of the target segmentation for training.")
    parser.add_argument("--num_tomo_crops", type=int, required=False, default = "16", help="Number of tomographic crops per training batch.")
    parser.add_argument("--reload_frequency", type=int, required=False, default = 15, help="Frequency of data reloading during training.")
    parser.add_argument("--lr", type=float, required=False, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, required=False, default=100, help="Number of epochs for training.")
    parser.add_argument("--val_interval", type=int, required=False, default=15, help="Number of epochs to wait prior to measuring validation metrics.")    
    parser.add_argument("--mlflow_tracking_uri", type=str, required=False, default = "http://mlflow.mlflow.svc.cluster.local:5000", help="MLflow tracking URI.")
    parser.add_argument("--mlflow_experiment_name", type=str, required=False, default = "model_train", help="Name of the MLflow experiment.")

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
        model_save_path=args.model_save_path,
        model_weights=args.model_weights,
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,
        num_tomo_crops=args.num_tomo_crops,
        reload_frequency=args.reload_frequency,
        lr=args.lr,
        num_epochs=args.num_epochs,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )

if __name__ == "__main__":
    cli()