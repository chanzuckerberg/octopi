from monai.networks.nets import UNet
import torch.nn as nn
import torch

class myUNet:
    def __init__(self, num_classes, device):

        self.device = device
        self.num_classes = num_classes
        
        # Placeholder for the model
        self.model = None

    def build_model(
        self, 
        channels=[32,64,128,128], 
        strides=[2,2,1], 
        num_res_units=1,
        dropout=0.0
        ):
        """Creates the Unet model based on provided parameters."""
        
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.num_classes,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout
        )
    
    def bayesian_search(self, trial):
        """Defines the Bayesian optimization search space and builds the model with suggested parameters."""
        
        # Define the search space
        num_layers = trial.suggest_int("num_layers", 3, 5)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
        num_res_units = trial.suggest_int("num_res_units", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        
        # Create channel sizes and strides
        downsampling_channels = [base_channel * (2 ** i) for i in range(num_layers)]
        hidden_channels = [downsampling_channels[-1]] * hidden_layers
        channels = downsampling_channels + hidden_channels
        strides = [2] * (num_layers - 1) + [1] * hidden_layers
        
        self.build_model(channels, strides, num_res_units, dropout)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return {
            'num_classes': self.num_classes,
            'channels': self.model.channels,
            'strides': self.model.strides,
            'num_res_units': self.model.num_res_units
        }
