from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch import io, trainer, utils, data
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import UNet
import torch, os, argparse
from typing import List, Optional

# model_save_path = '/mnt/simulations/ml_challenge/resunet_results'
# copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"
# segmentation_name = 'segmentation'
# my_channels = [32,64,128,128]
# my_strides = [2,2,1]
# my_num_res_units = 2
# my_num_samples = 16
# reload_frequency = 5
# lr = 1e-3

def train_model(
    copick_config_path: str,
    target_name: str = None,
    target_user_id: str = None,
    target_session_id: str = None,
    trainRunIDs: List[str] = None,
    validateRunIDs: List[str] = None,    
    channels: List[int] = [32,64,128,128],
    strides: List[int] = [2,2,1],
    res_units: int = 2,
    model_save_path: str = 'results',
    model_weights: Optional[str] = None,
    num_tomo_crops: int = 16,
    reload_frequency: int = 15,
    lr: float = 1e-3,
    num_epochs: int = 100,
    val_interval: int = 25,
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

    results = train.local_train(data_generator, 
                                model_save_path, 
                                max_epochs = num_epochs,
                                my_num_samples = num_tomo_crops,
                                reload_frequency = reload_frequency,
                                val_interval = val_interval,
                                verbose=True)

    parameters_save_name = os.path.join(model_save_path, 'training_parameters.json')
    io.save_parameters_to_json(model, train, data_generator, parameters_save_name)

    results_save_name = os.path.join(model_save_path, 'results.json')
    io.save_results_to_json(results, results_save_name)

# Entry point with argparse
def cli():
    """
    CLI entry point for training models where results are saved to a local directory.
    """

    parser = argparse.ArgumentParser(description="Train a UNet model on CryoET Tomograms.")
    parser.add_argument("--copick-config-path", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--trainRunIDs", type=utils.parse_list, required=False, help="List of training run IDs, e.g., run1,run2,run3 or [run1,run2,run3].")
    parser.add_argument("--validateRunIDs", type=utils.parse_list, required=False, help="List of validation run IDs, e.g., run4,run5,run6 or [run4,run5,run6].")
    parser.add_argument("--channels", type=utils.parse_int_list, required=False, help="List of channel sizes for each layer, e.g., 32,64,128,128 or [32,64,128,128].")
    parser.add_argument("--strides", type=utils.parse_int_list, required=False, help="List of stride sizes for each layer, e.g., 2,2,1 or [2,2,1].")
    parser.add_argument("--res-units", type=int, required=False, help="Number of residual units in the UNet.")
    parser.add_argument("--model-save-path", type=str, required=False, help="Path to save the trained model and results.")
    parser.add_argument("--target-name", type=str, required=False, help="Name of the target segmentation.")
    parser.add_argument("--target-user-id", type=str, required=False, help="User ID for the target segmentation.")
    parser.add_argument("--target-session-id", type=str, required=False, help="Session ID for the target segmentation.")
    parser.add_argument("--num-tomo-crops", type=int, required=False, help="Number of tomogram crops to use.")
    parser.add_argument("--reload-frequency", type=int, required=False, help="Frequency to reload training data.")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate for the optimizer.")
    parser.add_argument("--num-epochs", type=int, required=False, help="Number of training epochs.")
    parser.add_argument("--val-interval", type=int, required=False, default=25, help="Interval for validation metric calculations.")

    args = parser.parse_args()

    # Call the training function
    train_model(
        copick_config_path=args.copick_config_path,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
        channels=args.channels,
        strides=args.strides,
        res_units=args.res_units,
        model_save_path=args.model_save_path,
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,
        num_tomo_crops=args.num_tomo_crops,
        reload_frequency=args.reload_frequency,
        lr=args.lr,
        num_epochs=args.num_epochs,
        val_interval=args.val_interval,
    )

if __name__ == "__main__":
    cli()