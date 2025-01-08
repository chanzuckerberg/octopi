from monai.networks.nets import UNet, AttentionUnet
from model_explore import my_metrics, io, losses
from monai.losses import FocalLoss, TverskyLoss
from monai.metrics import ConfusionMatrixMetric
from model_explore.pytorch import trainer
from mlflow.tracking import MlflowClient
import torch, mlflow, os, optuna

def build_model(trial, data_generator, device):
    """Build the model with parameters suggested by Optuna."""

    # Loss Parameters
    loss_name = trial.suggest_categorical("loss_function", 
                                         ["FocalLoss", "TverskyLoss",'WeightedFocalTverskyLoss', 'FocalTverskyLoss'])
    if loss_name == "FocalLoss":
        gamma = round(trial.suggest_float("gamma", 0.1, 4), 3)
        loss_function = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=gamma)
    elif loss_name == "TverskyLoss":
        alpha = round(trial.suggest_float("alpha", 0.15, 0.85), 3)
        beta = 1.0 - alpha
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)
    elif loss_name == 'WeightedFocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 4), 3)
        alpha = round(trial.suggest_float("alpha", 0.15, 0.85), 3)
        beta = 1.0 - alpha
        weight_tversky = round(trial.suggest_float("weight_tversky", 0.1, 0.9), 3)
        weight_focal = 1.0 - weight_tversky
        loss_function = losses.WeightedFocalTverskyLoss(
            gamma=gamma, alpha=alpha, beta=beta,
            weight_tversky=weight_tversky, weight_focal=weight_focal
        )
    elif loss_name == 'FocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 4.0), 3)
        alpha = round(trial.suggest_float("alpha", 0.15, 0.85), 3)
        beta = 1.0 - alpha
        loss_function = losses.FocalTverskyLoss(gamma=gamma, alpha=alpha, beta=beta)

    # Model parameters
    num_layers = trial.suggest_int("num_layers", 3, 5)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
    base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])

    # Create channel sizes and strides
    downsampling_channels = [base_channel * (2 ** i) for i in range(num_layers)]
    hidden_channels = [downsampling_channels[-1]] * hidden_layers
    channels = downsampling_channels + hidden_channels
    strides = [2] * (num_layers - 1) + [1] * hidden_layers

    # Create the model
    model_type = trial.suggest_categorical("model_type", ["UNet", "AttentionUnet"])
    if model_type == "UNet":    num_res_units = trial.suggest_int("num_res_units", 1, 3)
    else:                       num_res_units = 0

    # # Calculate channels and strides
    # channels = [base_channel * (2 ** min(i, num_layers - 2)) for i in range(num_layers)]
    # strides_pattern = [2 if channels[i] != channels[i + 1] else 1 for i in range(len(channels) - 1)]

    # Create the model
    Nclass = data_generator.Nclasses
    model = create_model(model_type, Nclass, channels, strides, num_res_units, device)

    # Define metrics
    metrics_function = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["recall", "precision", "f1 score"],
        reduction="none"
    )

    # Sample crop size and num_samples
    samplings = {
        'crop_size': trial.suggest_int("crop_size", 64, 160, step=16),
        'num_samples': 8
    }

    return model, loss_function, metrics_function, samplings

def objective(
    trial, 
    epochs, 
    device, 
    data_generator, 
    best_metric='avg_f1', 
    random_seed=42, 
    val_interval=15):
    
    # utils.set_seed(random_seed)

    # Set a unique run name for each trial
    trial_num = f"trial_{trial.number}"

    # Start a new MLflow run for each trial
    with mlflow.start_run(run_name=trial_num, nested=True):

        # Build model
        model, loss_function, metrics_function, samplings = build_model(trial, data_generator, device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create trainer
        model_trainer = trainer.unet(model, device, loss_function, metrics_function, optimizer)

        # Sample crop size and train model
        score = train_model(trial, model_trainer, data_generator, epochs, val_interval, 
                            samplings['crop_size'], samplings['num_samples'], best_metric)

        # Log parameters and metrics
        params = {
            'model': io.get_model_parameters(model),
            'optimizer': io.get_optimizer_parameters(model_trainer)
        }
        mlflow.log_params(io.flatten_params(params))

        # Save best model and associated parameters
        best_score_so_far = get_best_score(trial)        
        if score > best_score_so_far:
            torch.save(model_trainer.model.state_dict(), 'model_exploration/best_metric_model.pth')
            io.save_parameters_to_json(model, model_trainer, data_generator, 'model_exploration/training_parameters.json')        

        return score      


def multi_gpu_objective(parent_run_id, trial, epochs, data_generator, random_seed=42, val_interval=5, best_metric='avg_f1', gpu_count=1):
    
    # utils.set_seed(random_seed)

    # Set up the MLflow run and GPU device for the trial
    device, client, target_run_id = setup_parallel_trial_run(trial, parent_run_id, gpu_count)
    
    # Build model
    model, loss_function, metrics_function, samplings = build_model(trial, data_generator, device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create trainer
    model_trainer = trainer.unet(model, device, loss_function, metrics_function, optimizer)
    model_trainer.set_parallel_mlflow(client, target_run_id)

    # Sample crop size and train model
    score = train_model(trial, model_trainer, data_generator, epochs, val_interval, 
                        samplings['crop_size'], samplings['num_samples'], best_metric)

    # Log training parameters.
    params = {
        'model': io.get_model_parameters(model),
        'optimizer': io.get_optimizer_parameters(model_trainer)
    }    
    my_metrics.my_log_param(io.flatten_params(params), client = client, trial_run_id = target_run_id )

    # Save best model and associated parameters
    best_score_so_far = get_best_score(trial)
    if score > best_score_so_far:
        torch.save(model_trainer.model.state_dict(), 'model_exploration/best_metric_model.pth')
        io.save_parameters_to_json(model, model_trainer, data_generator, 'model_exploration/training_parameters.json')      

    return score    

    # except optuna.TrialPruned:
    #     print(f"[Trial {trial_num}] Pruned due to failure or invalid score.")
    #     raise  # Let Optuna handle pruned trials

def get_best_score(trial):
    """
    Get the best score from the trial - default to 0 for first trial.
    """
    try:               return trial.study.best_value
    except ValueError: return 0

##############################################################################################################################

def setup_parallel_trial_run(trial, parent_run_id=None, gpu_count=1):
    """
    Set up the MLflow run and GPU device for a trial.
    """
    trial_num = f"trial_{trial.number}"

    # Multi-GPU scenario with a parent run
    mlflow_client = MlflowClient()
    trial_run = mlflow_client.create_run(
        experiment_id=mlflow_client.get_run(parent_run_id).info.experiment_id,
        tags={"mlflow.parentRunId": parent_run_id},
        run_name=trial_num
    )
    target_run_id = trial_run.info.run_id
    print(f"Logging trial {trial.number} data to MLflow run: {target_run_id}")

    # Assign GPU device
    if gpu_count > 1:
        gpu_id = trial.number % gpu_count
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device, mlflow_client, target_run_id

def train_model(trial, model_trainer, data_generator, epochs, val_interval, crop_size, num_samples, best_metric):
    """Train the model and return the score."""
    try:
        results = model_trainer.train(
            data_generator,
            model_save_path=None,
            crop_size=crop_size,
            max_epochs=epochs,
            val_interval=val_interval,
            my_num_samples=num_samples,
            best_metric=best_metric,
            use_mlflow=True,
            verbose=False
        )
        return results['best_metric']
    except torch.cuda.OutOfMemoryError:
        print(f"[Trial Failed] Out of Memory for crop_size={crop_size} and num_samples={num_samples}")
        trial.set_user_attr("out_of_memory", True)
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"[Trial Failed] Unexpected error: {e}")
        trial.set_user_attr("error", str(e))
        raise optuna.TrialPruned()    

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