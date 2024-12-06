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
    best_metric: str = 'avg_f1',
    random_seed: int = 42,
    val_interval: int = 15):

    utils.set_seed(random_seed)
        
    # Set a unique run name for each trial
    trial_num = f"trial_{trial.number}"

    # Start a new MLflow run for each trial
    with mlflow.start_run(run_name = trial_num, nested=True):  # Nested=True allows it to be part of the overall experiment run

        # Suggest alpha between 0.0 and 1.0 - Calculate beta based on the constraint alpha + beta = 1.0
        alpha = trial.suggest_float("alpha", 0.15, 0.85)
        beta = 1.0 - alpha

        # Monai Functions
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
        metrics_function = ConfusionMatrixMetric(include_background=False, 
                                                 metric_name=["recall",'precision','f1 score'], 
                                                 reduction="none",
                                                 )

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
        model_type = trial.suggest_categorical("model_type", ["UNet", "AttentionUnet"])
        model = create_model(model_type, Nclass, channels, strides_pattern, num_res_units, device)

        # Define your loss and optimizer, and return the objective (e.g., validation loss)
        lr = 1e-3
        # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)        
        optimizer = torch.optim.Adam(model.parameters(), lr)

        # Create UNet-Trainer
        model_trainer = trainer.unet(model, device, loss_function, metrics_function, optimizer)

        # Sample crop size in increments of 16
        # Will sample from [64, 80, 96, 112, 128, 144, 160]
        dim_in = trial.suggest_int("crop_size", 64, 160, step=16)  
        # num_samples = trial.suggest_int("num_samples", 4, 24, step=4)       
        num_samples = 8

        # Train the Model
        try:
            results = model_trainer.train(
                data_generator, 
                model_save_path = None,
                crop_size = dim_in,
                max_epochs = epochs,
                val_interval = val_interval,
                my_num_samples = num_samples,
                use_mlflow = True, verbose=False )
            score = results['best_metric']
        except torch.cuda.OutOfMemoryError:
            print(f"[Trial Failed] Out of Memory for crop_size={dim_in} and num_samples={num_samples}")
            trial.set_user_attr("out_of_memory", True)  # Optional: Log this for analysis
            return float("inf")  # Indicate failure for this trial

        # Optional: Save Model if Best Score
        try: 
            if score > trial.study.best_value:
                torch.save( model_trainer.model.state_dict('model_exploration/best_metric_model.pth') )
                model_trainer.plot_results(save_plot='model_exploration/net_train_history.png')
        except:
            # We need to have a score available to measure performance
            pass
        
        # Log training parameters.
        params = {
            'model': io.get_model_parameters(model),
            'optimizer': io.get_optimizer_parameters(model_trainer)
        }
        mlflow.log_params(io.flatten_params(params))

        return score


##############################################################################################################################

def multi_gpu_objective(parent_run_id,
                        trial,  
                        epochs,
                        data_generator,
                        random_seed = 42,
                        val_interval: int = 5,
                        gpu_count = 1,):
    
    utils.set_seed(random_seed)

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

    # Asign each trial to a specific GPU based on the trial number
    if gpu_count > 1:
        gpu_id = trial.number % gpu_count  # Cycle through available GPUs
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)  # Set the current GPU for this trial
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Suggest alpha between 0.0 and 1.0 - Calculate beta based on the constraint alpha + beta = 1.0
    # alpha = trial.suggest_float("alpha", 0.15, 0.85)
    # beta = 1.0 - alpha

    alpha = 0.24
    beta = 1.0 - alpha

    # Monai Functions
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)  
    metrics_function = ConfusionMatrixMetric(include_background=False, 
                                                metric_name=["recall",'precision','f1 score'], 
                                                reduction="none",
                                                )        
        
    # Sample number of channels
    # num_layers = trial.suggest_int("num_layers", 3, 6)
    
    # # Generate increasing channels based on 16, 32, 64
    # base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
    # channels = [base_channel * (2 ** min(i, num_layers - 2)) for i in range(num_layers)]
    
    num_layers = 4
    base_channel = 16
    channels = [base_channel * (2 ** min(i, num_layers - 2)) for i in range(num_layers)]

    # Generate strides with exact control for num_downsampling_layers
    strides_pattern = []
    for i in range(len(channels)-1):
        if channels[i] != channels[i+1]:   strides_pattern.append(2)
        else:                              strides_pattern.append(1)

    # Number of residual units
    # num_res_units = trial.suggest_int("num_res_units", 1, 3)
    num_res_units = 2

    # Now use channels, strides, and num_res_units in your model definition
    Nclass = data_generator.Nclasses
    model_type = trial.suggest_categorical("model_type", ["UNet", "AttentionUnet"])
    model = create_model(model_type, Nclass, channels, strides_pattern, num_res_units, device)

    # Define your loss and optimizer, and return the objective (e.g., validation loss)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    # Create UNet-Trainer
    train = trainer.unet(model, device, loss_function, metrics_function, optimizer)

    # Sample crop size in increments of 16
    # Will sample from [64, 80, 96, 112, 128, 144, 160]
    dim_in = trial.suggest_int("crop_size", 64, 160, step=16)  
    num_samples = trial.suggest_int("num_samples", 4, 16, step=4)     

    # Train the Model
    try:
        score = train.mlflow_train(data_generator, 
                                    crop_size = dim_in,
                                    max_epochs = epochs,
                                    val_interval = val_interval,
                                    my_num_samples = num_samples,
                                    verbose=True)[0]        
    except torch.cuda.OutOfMemoryError:
        print(f"[Trial Failed] Out of Memory for crop_size={dim_in} and num_samples={num_samples}")
        trial.set_user_attr("out_of_memory", True)  # Optional: Log this for analysis
        return float("inf")  # Indicate failure for this trial
    

    import pdb; pdb.set_trace() 

    # Log training parameters.
    params = {
        'model': io.get_model_parameters(model),
        'optimizer': io.get_optimizer_parameters(train)
    }    
    my_metrics.my_log_param(io.flatten_params(params), run_id = target_run_id,   )

    import pdb; pdb.set_trace()    
    
    # for param_key, param_value in params.items():
    #     mlflow_client.log_param(run_id = target_run_id, key=param_key, value=param_value)    

    # # Log the score (e.g., validation loss or F1 score) for each trial
    # my_metrics.my_log_metric('score', score, 0, mlflow_client, target_run_id)

    return score

##############################################################################################################################

def create_model(model_type, n_classes, channels, strides_pattern, num_res_units, device):
    """
    Create either a UNet or AttentionUnet model based on trial parameters.
    
    Args:
        trial: Optuna trial object
        n_classes: Number of output classes
        channels: List of channel sizes
        strides_pattern: List of stride values
        num_res_units: Number of residual units (only used for UNet)
        device: torch device to place model on
    """
    
    if model_type == "UNet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=channels,
            strides=strides_pattern,
            num_res_units=num_res_units,
        )
    else:  # AttentionUnet
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=channels,
            strides=strides_pattern,
        )
    
    return model.to(device)