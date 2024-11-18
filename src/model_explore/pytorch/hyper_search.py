from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.networks.nets import UNet, AttentionUnet
from model_explore.pytorch import utils, my_metrics
from monai.metrics import ConfusionMatrixMetric
from model_explore.pytorch import io, trainer
from mlflow.tracking import MlflowClient
import torch, mlflow


def objective(
    trial,  
    epochs,
    device,
    data_generator,
    num_samples: int = 16,
    reload_frequency: int = 15,
    random_seed: int = 42,
    val_interval: int = 25):

    utils.set_seed(random_seed)
        
    # Set a unique run name for each trial
    trial_num = f"trial_{trial.number}"

    # Start a new MLflow run for each trial
    with mlflow.start_run(run_name = trial_num, nested=True):  # Nested=True allows it to be part of the overall experiment run

        # Monai Functions
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  
        metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")

        # Sample number of channels
        num_layers = trial.suggest_int("num_layers", 3, 6)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)

        # Generate increasing channels based on 16, 32, 64
        base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
        channels = [base_channel * (2 ** min(i, num_layers - 2)) for i in range(num_layers)]
        
        # Generate strides with exact control for num_downsampling_layers
        strides_pattern = []
        for i in range(len(channels)-1):
            if channels[i] != channels[i+1]:   strides_pattern.append(2)
            else:                              strides_pattern.append(1)

        # Number of residual units
        num_res_units = trial.suggest_int("num_res_units", 1, 3)

        # Now use channels, strides, and num_res_units in your model definition
        Nclass = data_generator.Nclasses
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=Nclass,
            channels=channels,
            strides=strides_pattern,
            num_res_units=num_res_units,
        ).to(device)

        # model = AttentionUnet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=n_classes,
        #     channels=channels,
        #     strides=strides_pattern,
        # ).to(device)            

        # # Sample learning rate using Optuna
        # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)

        # Define your loss and optimizer, and return the objective (e.g., validation loss)
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr)

        # Create UNet-Trainer
        train = trainer.unet(model, device, loss_function, metrics_function, optimizer)

        # Train the Model
        score = train.mlflow_train(data_generator, 
                                    max_epochs = epochs,
                                    val_interval = val_interval,
                                    my_num_samples = num_samples,
                                    reload_frequency = reload_frequency,
                                    verbose=False)[0]
        
        # Log training parameters.
        params = {
            'model': io.get_model_parameters(model),
            'optimizer': io.get_optimizer_parameters(train)
        }
        mlflow.log_params(io.flatten_params(params))

        return score


##############################################################################################################################

def objective2(parent_run_id,
              trial,  
              n_classes,
              train_loader, 
              val_loader, 
              loss_function,
              metrics_function, 
              epochs,
              random_seed = 42,
              gpu_count = 1,):
    
    # Initialize MLflow client
    mlflow_client = MlflowClient()

    # Set a unique run name for each trial
    trial_num = f"trial_{trial.number}"

    # Start a new MLflow run for each trial
    # with mlflow.start_run(run_name = trial_num, 
    #                       nested=True,
    #                       tags={MLFLOW_PARENT_RUN_ID: parent_run_id}) as child_run:  # Nested=True allows it to be part of the overall experiment run
        
        # # Print the active run name and ID for debugging
        # print("Starting new MLflow run:", child_run.info.run_name, ", ", child_run.info.run_id)

    # Create a unique run for each trial using the parent experiment ID
    trial_run = mlflow_client.create_run(experiment_id=mlflow_client.get_run(parent_run_id).info.experiment_id,
                                         tags={"mlflow.parentRunId": parent_run_id}, run_name = trial_num)
    target_run_id = trial_run.info.run_id
    print(f"Logging trial {trial.number} data to MLflow run: {target_run_id}")

    utils.set_seed(random_seed)

    # Asign each trial to a specific GPU based on the trial number
    if gpu_count > 1:
        gpu_id = trial.number % gpu_count  # Cycle through available GPUs
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)  # Set the current GPU for this trial
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Sample number of channels
    num_layers = trial.suggest_int("num_layers", 3, 6)
    
    # Generate increasing channels based on 16, 32, 64
    base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
    channels = [base_channel * (2 ** min(i, num_layers - 2)) for i in range(num_layers)]
    
    # Generate strides with exact control for num_downsampling_layers
    strides_pattern = []
    for i in range(len(channels)-1):
        if channels[i] != channels[i+1]:   strides_pattern.append(2)
        else:                              strides_pattern.append(1)

    # Number of residual units
    num_res_units = trial.suggest_int("num_res_units", 1, 3)

    # mlflow.log_params({
    #     "channels": channels,
    #     "num_res_units": num_res_units
    # })

    params = {
        "channels": channels,
        "strides_pattern": strides_pattern,
        "num_res_units": num_res_units,
    }

    for param_key, param_value in params.items():
        mlflow_client.log_param(run_id = target_run_id, key=param_key, value=param_value)
        
    # Now use channels, strides, and num_res_units in your model definition
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_classes,
        channels=channels,
        strides=strides_pattern,
        num_res_units=num_res_units,
    ).to(device)

    # Define your loss and optimizer, and return the objective (e.g., validation loss)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)

    score = train(train_loader, val_loader, 
                    model, device, loss_function, 
                    metrics_function, 
                    optimizer, max_epochs=epochs,
                    client = mlflow_client,
                    trial_run_id = target_run_id,
                    my_device = device)

    # Log the score (e.g., validation loss or F1 score) for each trial
    my_metrics.my_log_metric('score', score, 0, mlflow_client, target_run_id)

    return objective

##############################################################################################################################

