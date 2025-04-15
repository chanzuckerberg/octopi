import torch.nn as nn
import torch

class myModelTemplate:
    def __init__(self):
        """
        Initialize the model template.
        """
        # Placeholder for the model and config
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """
        Build the model based on provided parameters in a config dictionary.
        """
        pass

    def bayesian_search(self, trial, num_classes: int):
        """
        Define the hyperparameter search space for Bayesian optimization and build the model.

        The search space below is just an example and can be customized.
        
        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            num_classes (int): Number of classes in the dataset.
        """
        pass
    
    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return self.config
