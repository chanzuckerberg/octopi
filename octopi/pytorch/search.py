from monai.metrics import ConfusionMatrixMetric
from mlflow.tracking import MlflowClient
from octopi.pytorch import trainer
from octopi.models import common
import torch, mlflow, optuna, gc
from octopi.utils import io

# --------------------------
# Run Model Search
# --------------------------

class ModelExplorer:
    def __init__(self, data_generator, model_type="Unet", output='explore_results'):
        """
        Class to handle model creation, training, and optimization.

        Args:
            data_generator (object): Data generator object containing dataset properties.
            model_type (str): Type of model to build ("UNet", "AttentionUnet").
            output (str): Directory to save output results.
        """
        self.data_generator = data_generator
        self.Nclasses = data_generator.Nclasses
        self.device = None
        self.model_type = model_type
        self.model = None
        self.loss_function = None
        self.metrics_function = None
        self.sampling = None
        
        # Define results directory path
        self.output = output               

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
            'crop_size': trial.suggest_int("crop_size", 48, 192, step=16),
            'num_samples': 16
        }
        self.config['dim_in'] = self.sampling['crop_size']

    def _define_optimizer(self, trial):
        # Define optimizer
        # lr0 = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        # wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)      

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
                verbose=False,
                trial=trial
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
        with mlflow.start_run(run_name=trial_num):
            
            # Build model
            self.my_build_model(trial)

            # Create trainer
            self._define_optimizer(trial)
            model_trainer = trainer.ModelTrainer(
                self.model, self.device, self.loss_function, 
                self.metrics_function, self.optimizer
            )

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

            # Save best model
            self._save_best_model(trial, model_trainer, score)

            # Cleanup
            self.cleanup(model_trainer)

        return score

    def get_best_score(self, trial):
        """Retrieve the best score from the trial."""
        try:
            return trial.study.best_value
        except ValueError:
            return -float('inf')

    def cleanup(self, model_trainer):
        """Handles cleanup of resources."""

        # Log training parameters
        params = {
            'model': self.model_builder.get_model_parameters(),
            'optimizer': io.get_optimizer_parameters(model_trainer)
        }    
        model_trainer.my_log_params(io.flatten_params(params))
    
        # Delete the trainer and optimizer objects
        del model_trainer, self.optimizer

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
