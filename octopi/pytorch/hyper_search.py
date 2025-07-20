from monai.metrics import ConfusionMatrixMetric
from octopi.pytorch import trainer
from mlflow.tracking import MlflowClient
from octopi.models import common
import torch, mlflow, optuna, gc
from octopi.utils import io

class BayesianModelSearch:

    def __init__(self, data_generator, model_type="Unet", parent_run_id=None, parent_run_name=None):
        """
        Class to handle model creation, training, and optimization.

        Args:
            data_generator (object): Data generator object containing dataset properties.
            model_type (str): Type of model to build ("UNet", "AttentionUnet").
        """
        self.data_generator = data_generator
        self.Nclasses = data_generator.Nclasses
        self.device = None
        self.model_type = model_type
        self.model = None
        self.loss_function = None
        self.metrics_function = None
        self.sampling = None
        self.parent_run_id = parent_run_id
        
        # Define results directory path
        self.results_dir = f'explore_results_{self.model_type}'            

    def my_build_model(self, trial):
        """Builds and initializes a model based on Optuna-suggested parameters."""
       
        # Build the model
        self.model_builder = common.get_model(self.model_type)
        self.model_builder.bayesian_search(trial, self.Nclasses)
        self.model = self.model_builder.model.to(self.device)
        self.config = self.model_builder.config

        # Define loss function
        self.loss_function = common.get_loss_function(trial)

        # Define metrics
        self.metrics_function = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["recall", "precision", "f1 score"],
            reduction="none"
        )

        # Sample crop size and num_samples
        self.sampling = {
            'crop_size': trial.suggest_int("crop_size", 48, 160, step=16),
            'num_samples': 8
        }
        self.config['dim_in'] = self.sampling['crop_size']

    def _define_optimizer(self):
        # Define optimizer
        lr0 = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0, weight_decay=1e-5)

    def _train_model(self, trial, model_trainer, epochs, val_interval, crop_size, num_samples, best_metric):
        """Handles model training and error handling."""
        try:
            results = model_trainer.train(
                self.data_generator,
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
            print(f"[Trial Failed] OOM Error for model={model_trainer.model}, crop_size={crop_size}, num_samples={num_samples}")
            trial.set_user_attr("out_of_memory", True)
            raise optuna.TrialPruned()

        except Exception as e:
            print(f"[Trial Failed] Unexpected error: {e}")
            trial.set_user_attr("error", str(e))
            raise optuna.TrialPruned()        

    def objective(self, trial, epochs, device, val_interval=15, best_metric="avg_f1"):
        """Runs the full training process for a given trial."""

        # Set device
        self.device = device
        
        # Set a unique run name for each trial
        trial_num = f"trial_{trial.number}"

        # Start MLflow run
        with mlflow.start_run(run_name=trial_num, nested=True):
            
            # Build model
            self.my_build_model(trial)

            # Create trainer
            self._define_optimizer()
            model_trainer = trainer.ModelTrainer(self.model, self.device, self.loss_function, self.metrics_function, self.optimizer)

            # Train model and evaluate score
            score = self._train_model(
                trial, model_trainer, epochs, val_interval, 
                self.sampling['crop_size'], self.sampling['num_samples'], 
                best_metric)               

            # Log parameters and metrics
            params = {
                'model': self.model_builder.get_model_parameters(),
                'optimizer': io.get_optimizer_parameters(model_trainer)
            }
            model_trainer.my_log_params(io.flatten_params(params))

            # Explicitly set the parent run ID
            mlflow.log_param("parent_run_id", self.parent_run_id)
            mlflow.log_param("parent_run_name", self.parent_run_name)         

            # Save best model
            self._save_best_model(trial, model_trainer, score)

            # Cleanup
            self.cleanup(model_trainer, self.optimizer)

        return score        

    def _setup_parallel_trial_run(self, trial, parent_run=None, gpu_count=1):
        """Set up parallel MLflow runs and assign GPU for the trial."""
        
        trial_num = f"trial_{trial.number}"

        # Create a child run under the parent MLflow experiment
        mlflow_client = MlflowClient()
        self.trial_run = mlflow_client.create_run(
            experiment_id=mlflow_client.get_run(parent_run.info.run_id).info.experiment_id,
            tags={"mlflow.parentRunId": parent_run.info.run_id},
            run_name=trial_num
        )
        target_run_id = self.trial_run.info.run_id
        print(f"Logging trial {trial.number} data to MLflow run: {target_run_id}")

        # Assign GPU device
        if gpu_count > 1:
            gpu_id = trial.number % gpu_count
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return device, mlflow_client, target_run_id
    
    def multi_gpu_objective(self, parent_run, trial, epochs, val_interval=5, best_metric="avg_f1", gpu_count=1):
        """
        Trains model on multiple GPUs using a parent MLflow run.
        """
        self.device, self.client, self.target_run_id = self._setup_parallel_trial_run(trial, parent_run, gpu_count)
        
        # Build model
        self.my_build_model(trial)

        # Create trainer
        self._define_optimizer()
        model_trainer = trainer.ModelTrainer(self.model, self.device, self.loss_function, self.metrics_function, self.optimizer)
        model_trainer.set_parallel_mlflow(self.client, self.target_run_id)

        # # Train model
        # score = self._train_model(
        #     trial, model_trainer, epochs, val_interval, 
        #     self.sampling['crop_size'], self.sampling['num_samples'], 
        #     best_metric)

        # Train Model, with error handling
        try:
            score = self._train_model(
                trial, model_trainer, epochs, val_interval, 
                self.sampling['crop_size'], self.sampling['num_samples'], 
                best_metric)
        except Exception as e:
            print(f"[Trial Failed] Unexpected error: {e}")
            trial.set_user_attr("error", str(e))
            raise optuna.TrialPruned()
        
        # Log training parameters
        params = {
            'model': self.model_builder.get_model_parameters(),
            'optimizer': io.get_optimizer_parameters(model_trainer)
        }    
        model_trainer.my_log_params(io.flatten_params(params))
        model_trainer.my_log_params({"parent_run_name": parent_run.info.run_name})

        # Save best model
        self._save_best_model(trial, model_trainer, score)

        # Cleanup
        self.cleanup(model_trainer, self.optimizer)
        return score

    def _save_best_model(self, trial, model_trainer, score):
        """Saves the best model if it improves upon previous scores."""
        best_score_so_far = self.get_best_score(trial)
        if score > best_score_so_far:
            torch.save(model_trainer.model_weights, f'{self.results_dir}/best_model.pth')
            io.save_parameters_to_yaml(self.model_builder, model_trainer, self.data_generator, 
                                    f'{self.results_dir}/model_config.yaml')

    def get_best_score(self, trial):
        """Retrieve the best score from the trial."""
        try:
            return trial.study.best_value
        except ValueError:
            return 0

    def cleanup(self, model_trainer, optimizer):
        """Handles cleanup of resources."""

        # Delete the trainer and optimizer objects
        del model_trainer, optimizer

        # If the model object holds GPU memory, delete it explicitly and set it to None
        if hasattr(self, "model"):
            del self.model
            self.model = None

        # Optional: If your model_builder or other objects hold GPU references, delete them too
        if hasattr(self, "model_builder"):
            del self.model_builder
            self.model_builder = None

        # Clear the CUDA cache and force garbage collection
        torch.cuda.empty_cache()
        gc.collect()

        # Try Terminating Multi-GPU Runs, if not run current run (works for single GPU runs)
        try:
            self.client.set_terminated(self.target_run_id, status="FINISHED")
        except:
            mlflow.end_run()
