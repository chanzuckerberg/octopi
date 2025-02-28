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

    def build_model(self):
        """
        Build the model based on provided parameters.

        Example Args:
            channels (list of int): List defining the number of filters at each stage.
            strides (list of int): List defining the downsampling factors.
            num_res_units (int): Number of residual units per stage.
            dropout (float): Dropout rate.
        """
        pass

    def bayesian_search(self, trial):
        """
        Define the hyperparameter search space for Bayesian optimization and build the model.

        The search space below is just an example and can be customized.
        
        Args:
            trial (optuna.trial.Trial): Optuna trial object.
        """
        pass
    
    def get_model_parameters(self):
        """
        Retrieve stored model parameters for logging or analysis.

        Returns:
            dict: A dictionary of the current model parameters.
        """
        pass