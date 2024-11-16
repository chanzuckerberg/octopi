from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch import io, trainer, utils, data
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import UNet
import torch, mlflow, os
from typing import List

# model_save_path = '/mnt/simulations/ml_challenge/resunet_results'
# copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"
# segmentation_name = 'segmentation'
# my_channels = [32,64,128,128]
# my_strides = [2,2,1]
# my_num_res_units = 2
# my_num_samples = 16
# reload_frequency = 5
# lr = 1e-3

def train_model_on_mflow(
    copick_config_path: str,
    trainRunIDs: List[str],
    validateRunIDs: List[str],    
    channels: List[int],
    strides: List[int],
    res_units: int,
    model_save_path: str,
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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Create UNet-Trainer
    train = trainer.unet(model, device, loss_function, metrics_function, optimizer)

    # Set up ML-Flow
    utils.mlflow_setup()

    # Log ML-Flow Refrence and Start Training
    mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_experiment('fake-search')

    with mlflow.start_run():

        train.mlflow_train(data_generator, 
                        max_epochs = num_epochs,
                        my_num_samples = num_tomo_crops,
                        val_interval = 5,
                        reload_frequency = reload_frequency,
                        model_save_path = 'model_save_path',
                        verbose=False )
        
        # Log training parameters.
        params = {
            'model': io.get_model_parameters(model),
            'optimizer': io.get_optimizer_parameters(train),
            'dataloader': data_generator.get_dataloader_parameters()
        }
        mlflow.log_params(io.flatten_params(params))

    mlflow.end_run()
