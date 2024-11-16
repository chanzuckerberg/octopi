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
    trainRunIDs: List[str],
    validateRunIDs: List[str],    
    channels: List[int],
    strides: List[int],
    res_units: int,
    model_save_path: str,
    model_weights: Optional[str],
    target_name: str,
    target_user_id: str,
    target_session_id: str,
    num_tomo_crops: int,
    reload_frequency: int,
    lr: float,
    num_epochs: int
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
                                val_interval = 5,
                                verbose=True)

    parameters_save_name = os.path.join(model_save_path, 'training_parameters.json')
    io.save_parameters_to_json(model, train, data_generator, parameters_save_name)

    results_save_name = os.path.join(model_save_path, 'results.json')
    io.save_results_to_json(results, results_save_name)

# Entry point with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet model on tomographic data.")
    parser.add_argument("copick_config_path", type=str, help="Path to the CoPick configuration file.")
    parser.add_argument("--trainRunIDs", type=str, required=True, help="Comma-separated list of training run IDs.")
    parser.add_argument("--validateRunIDs", type=str, required=True, help="Comma-separated list of validation run IDs.")
    parser.add_argument("--channels", type=str, required=True, help="Comma-separated list of channel sizes for each layer.")
    parser.add_argument("--strides", type=str, required=True, help="Comma-separated list of stride sizes for each layer.")
    parser.add_argument("--res_units", type=int, required=True, help="Number of residual units in the UNet.")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model and results.")
    parser.add_argument("--target_name", type=str, required=True, help="Name of the target segmentation.")
    parser.add_argument("--target_user_id", type=str, required=True, help="User ID for the target segmentation.")
    parser.add_argument("--target_session_id", type=str, required=True, help="Session ID for the target segmentation.")
    parser.add_argument("--num_tomo_crops", type=int, required=True, help="Number of tomogram crops to use.")
    parser.add_argument("--reload_frequency", type=int, required=True, help="Frequency to reload training data.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs.")

    args = parser.parse_args()

    # Convert comma-separated arguments into lists
    trainRunIDs = args.trainRunIDs.split(",")
    validateRunIDs = args.validateRunIDs.split(",")
    channels = list(map(int, args.channels.split(",")))
    strides = list(map(int, args.strides.split(",")))

    # Ensure the model save path exists
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    # Call the training function
    train_model(
        copick_config_path=args.copick_config_path,
        trainRunIDs=trainRunIDs,
        validateRunIDs=validateRunIDs,
        channels=channels,
        strides=strides,
        res_units=args.res_units,
        model_save_path=args.model_save_path,
        target_name=args.target_name,
        target_user_id=args.target_user_id,
        target_session_id=args.target_session_id,
        num_tomo_crops=args.num_tomo_crops,
        reload_frequency=args.reload_frequency,
        lr=args.lr,
        num_epochs=args.num_epochs,
    )