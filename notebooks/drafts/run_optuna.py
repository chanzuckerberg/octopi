from model_explore.pytorch import io, data, hyper_search, utils
import torch, mlflow, os, copick, optuna

copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"
optuna_storage_path = "/mnt/simulations/ml_challenge/optuna_results"
experimental_mlflow_name = 'fake-search'
segmentation_name = 'segmentation'
num_epochs = 25
num_trials = 10

# Split Experiment into Train and Validation Runs
Nclass = io.get_num_classes(copick_config_path)
data_generator = data.train_generator(copick_config_path, 
                                      segmentation_name, 
                                      Nclasses = Nclass,
                                      tomo_batch_size = 20)
myRunIDs = data_generator.get_data_splits()

# Determine Device to Run Optuna On
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pruning = True
pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    )

# Explicitly initialize the TPE sampler
tpe_sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,     # Number of initial random trials before TPE kicks in
    n_ei_candidates=24,      # Number of candidate samples for Expected Improvement
    multivariate=True        # Use multivariate TPE for correlated parameter sampling
)

# Set up ML-Flow
utils.mlflow_setup()

print(f'Running Architecture Search Over 1 GPU\n')
mlflow.set_tracking_uri('http://mlflow.mlflow.svc.cluster.local:5000')
mlflow.set_experiment(experimental_mlflow_name)

storage = "sqlite:///example.db"
with mlflow.start_run():

    study = optuna.create_study(storage=storage,
                                direction="maximize",
                                load_if_exists=True,
                                pruner=pruner)

    mlflow.log_params({"random_seed": 42})
    mlflow.log_params(data_generator.get_dataloader_parameters())

    study.optimize(lambda trial: hyper_search.objective(trial, num_epochs, device, data_generator), 
                    n_trials=num_trials) 

print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_params}")