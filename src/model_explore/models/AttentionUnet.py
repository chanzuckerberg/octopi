from monai.networks.nets import AttentionUnet
import torch.nn as nn
import torch

class myAttentionUnet:
    def __init__(self):

        # Placeholder for the model and config
        self.model = None
        self.config = None

    def build_model( self, config: dict ):
        """Creates the AttentionUnet model based on provided parameters."""
        
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=config['num_classes'],
            channels=config['channels'],
            strides=config['strides'],
            dropout=config['dropout']
        )
        
        return self.model
    
    def bayesian_search(self, trial, num_classes: int):
        """Defines the Bayesian optimization search space and builds the model with suggested parameters."""
        
        # Define the search space
        num_layers = trial.suggest_int("num_layers", 3, 5)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
        
        # Create channel sizes and strides
        downsampling_channels = [base_channel * (2 ** i) for i in range(num_layers)]
        hidden_channels = [downsampling_channels[-1]] * hidden_layers
        channels = downsampling_channels + hidden_channels
        strides = [2] * (num_layers - 1) + [1] * hidden_layers
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
        
        config = {
            'architecture': 'AttentionUnet',
            'num_classes': num_classes,
            'channels': channels,
            'strides': strides,
            'dropout': dropout
        }

        return self.build_model(config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return self.config