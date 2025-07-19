from octopi.datasets import generators, multi_config_generator
from octopi.utils import config, parsers
from octopi.pytorch import hyper_search
import torch, mlflow, optuna
from typing import List
import pandas as pd

class ModelSearchSubmit:
    def __init__(
        self,
        copick_config: str,
        target_name: str,
        target_user_id: str,
        target_session_id: str,
        tomo_algorithm: str,
        voxel_size: float,
        Nclass: int,
        model_type: str,
        best_metric: str = 'avg_f1',
        num_epochs: int = 1000,
        num_trials: int = 100,
        data_split: str = 0.8,
        random_seed: int = 42,
        val_interval: int = 10,
        tomo_batch_size: int = 15,
        trainRunIDs: List[str] = None,
        validateRunIDs: List[str] = None,
        mlflow_experiment_name: str = 'explore',
    ):
        """
        Initialize the ModelSearch class for architecture search with Optuna.

        Args:
            copick_config (str or dict): Path to the CoPick configuration file or a dictionary for multi-config training.
            target_name (str): Name of the target for segmentation.
            target_user_id (str): Optional user ID for tracking.
            target_session_id (str): Optional session ID for tracking.
            tomo_algorithm (str): Tomogram algorithm to use.
            voxel_size (float): Voxel size for tomograms.
            Nclass (int): Number of prediction classes.
            model_type (str): Type of model to use.
            mlflow_experiment_name (str): MLflow experiment name.
            random_seed (int): Seed for reproducibility.
            num_epochs (int): Number of epochs per trial.
            num_trials (int): Number of trials for hyperparameter optimization.
            tomo_batch_size (int): Batch size for tomogram loading.
            best_metric (str): Metric to optimize.
            val_interval (int): Validation interval.
            trainRunIDs (List[str]): List of training run IDs.
            validateRunIDs (List[str]): List of validation run IDs.
            data_split (str): Data split ratios.
        """

        # Input parameters 
        self.copick_config = copick_config
        self.target_name = target_name
        self.target_user_id = target_user_id
        self.target_session_id = target_session_id
        self.tomo_algorithm = tomo_algorithm
        self.voxel_size = voxel_size
        self.Nclass = Nclass
        self.model_type = model_type
        self.mlflow_experiment_name = mlflow_experiment_name
        self.random_seed = random_seed
        self.num_epochs = num_epochs
        self.num_trials = num_trials
        self.tomo_batch_size = tomo_batch_size
        self.best_metric = best_metric
        self.val_interval = val_interval
        self.trainRunIDs = trainRunIDs
        self.validateRunIDs = validateRunIDs
        self.data_split = data_split
        
        # Data generator - will be initialized in _initialize_data_generator()
        self.data_generator = None

        # Set random seed for reproducibility
        config.set_seed(self.random_seed)

        # Initialize dataset generator
        self._initialize_data_generator()

    def _initialize_data_generator(self):
        """Initializes the data generator for training and validation datasets."""
        self._print_input_configs()

        if isinstance(self.copick_config, dict):
            self.data_generator = multi_config_generator.MultiConfigTrainLoaderManager(
                self.copick_config,
                self.target_name,
                target_session_id=self.target_session_id,
                target_user_id=self.target_user_id,
                tomo_algorithm=self.tomo_algorithm,
                voxel_size=self.voxel_size,
                Nclasses=self.Nclass,
                tomo_batch_size=self.tomo_batch_size
            )
        else:
            self.data_generator = generators.TrainLoaderManager(
                self.copick_config,
                self.target_name,
                target_session_id=self.target_session_id,
                target_user_id=self.target_user_id,
                tomo_algorithm=self.tomo_algorithm,
                voxel_size=self.voxel_size,
                Nclasses=self.Nclass,
                tomo_batch_size=self.tomo_batch_size
            )

        # Split datasets into training and validation
        ratios = parsers.parse_data_split(self.data_split)
        self.data_generator.get_data_splits(
            trainRunIDs=self.trainRunIDs,
            validateRunIDs=self.validateRunIDs,
            train_ratio = ratios[0], val_ratio = ratios[1], test_ratio = ratios[2],
            create_test_dataset = False
        )
        
        # Get the reload frequency
        self.data_generator.get_reload_frequency(self.num_epochs)
        
    def _print_input_configs(self):
        """Prints training configuration for debugging purposes."""
        print(f'\nTraining with:')
        if isinstance(self.copick_config, dict):
            for session, config in self.copick_config.items():
                print(f'  {session}: {config}')
        else:
            print(f'  {self.copick_config}')
        print()

    def run_model_search(self):
        """Performs model architecture search using Optuna and MLflow."""

        # Set up MLflow tracking
        try:
            tracking_uri = config.mlflow_setup()
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as e:
            print(f'Failed to set up MLflow tracking: {e}')
            pass

        mlflow.set_experiment(self.mlflow_experiment_name)
    
        # Create a storage object with heartbeat configuration
        storage_url = f"sqlite:///explore_results_{self.model_type}/trials.db"
        self.storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,  # Record heartbeat every minute
            grace_period=600,       # 10 minutes grace period
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1)
        )

        # Detect GPU availability
        gpu_count = torch.cuda.device_count()
        print(f'Running Architecture Search Over {gpu_count} GPUs\n')

        # Initialize model search object
        if gpu_count > 1:
            self._multi_gpu_optuna(gpu_count)
        else:
            model_search = hyper_search.BayesianModelSearch(self.data_generator, self.model_type)
            self._single_gpu_optuna(model_search)

    def _single_gpu_optuna(self, model_search):
        """Runs Optuna optimization on a single GPU."""
        
        with mlflow.start_run(nested=False) as parent_run:

            model_search.parent_run_id = parent_run.info.run_id
            model_search.parent_run_name = parent_run.info.run_name

            # Log the experiment parameters
            mlflow.log_params({"random_seed": self.random_seed})
            mlflow.log_params(self.data_generator.get_dataloader_parameters())
            mlflow.log_params({"parent_run_name": parent_run.info.run_name})

            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Create the study and run the optimization
            study = self.get_optuna_study()
            study.optimize(
                lambda trial: model_search.objective(
                    trial, self.num_epochs, device,
                    val_interval=self.val_interval,
                    best_metric=self.best_metric,
                ),
                n_trials=self.num_trials
            )

            # Save contour plot
            self.save_contour_plot_as_png(study)

            print(f"Best trial: {study.best_trial.value}")
            print(f"Best params: {study.best_params}")

    def _multi_gpu_optuna(self, gpu_count):
        """Runs Optuna optimization on multiple GPUs."""
        with mlflow.start_run() as parent_run:

            # Log the experiment parameters
            mlflow.log_params({"random_seed": self.random_seed})
            mlflow.log_params(self.data_generator.get_dataloader_parameters())
            mlflow.log_params({"parent_run_name": parent_run.info.run_name})

            # Run multi-GPU optimization
            study = self.get_optuna_study()
            study.optimize(
                lambda trial: hyper_search.BayesianModelSearch(self.data_generator, self.model_type).multi_gpu_objective(
                    parent_run, trial,
                    self.num_epochs,
                    best_metric=self.best_metric,
                    val_interval=self.val_interval,
                    gpu_count=gpu_count
                ),
                n_trials=self.num_trials,
                n_jobs=gpu_count
            )

            # Save contour Plot
            self.save_contour_plot_as_png(study)

        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_params}")

    def _get_optuna_sampler(self):
        """Returns Optuna's TPE sampler with default settings."""
        return optuna.samplers.TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True
        )
        # return optuna.samplers.BoTorchSampler(
        #     n_startup_trials=10,
        #     multivariate=True
        # )

    def get_optuna_study(self):
        """Returns the Optuna study object."""
        return optuna.create_study(
                storage=self.storage,
                direction="maximize",
                sampler=self._get_optuna_sampler(),
                load_if_exists=True,
                pruner=self._get_optuna_pruner()
            )

    def _get_optuna_pruner(self):
        """Returns Optuna's pruning strategy."""
        return optuna.pruners.MedianPruner()

    def save_contour_plot_as_png(self, study):
        """
        Save the contour plot of hyperparameter interactions as a PNG, 
        automatically extracting parameter names from the study object.

        Args:
            study: The Optuna study object.
            output_path: Path to save the PNG file.
        """
        # Extract all parameter names from the study trials
        all_params = set()
        for trial in study.trials:
            all_params.update(trial.params.keys())
        all_params = list(all_params)  # Convert to a sorted list for consistency

        # Generate the contour plot
        fig = optuna.visualization.plot_contour(study, params=all_params) 

        # Adjust figure size and font size
        fig.update_layout(
            width=6000, height=6000,  # Large figure size
            font=dict(size=40)  # Increase font size for better readability
        )

        # Save the plot as a PNG file
        fig.write_image(f'explore_results_{self.model_type}/contour_plot.png', scale=1)  

        # Extract trial data
        trials = [
            {**trial.params, 'objective_value': trial.value}
            for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        # Convert to DataFrame
        df = pd.DataFrame(trials)

        # Save to CSV
        df.to_csv(f"explore_results_{self.model_type}/optuna_results.csv", index=False)
