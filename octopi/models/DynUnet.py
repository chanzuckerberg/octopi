from monai.networks.nets import SegResNetDS
import torch.nn as nn
import torch

class mySegResNet:
    def __init__(self):
        # Placeholder for the model and config
        self.model = None
        self.config = None

    def build_model(
        self,
        config: dict
    ):
        """
        Creates the SegResNetDS model based on provided parameters.
        
        Args:
            init_filters (int): Number of output channels for the initial convolution.
            blocks_down (tuple): Tuple defining the number of blocks at each downsampling stage.
            dsdepth (int): Depth for deep supervision (number of output scales).
            act (str): Activation type.
            norm (str): Normalization type.
            blocks_up (tuple or None): Number of upsample blocks (if None, uses default behavior).
            upsample_mode (str): Upsampling mode, e.g. 'deconv' or 'trilinear'.
            resolution (optional): If provided, adjusts non-isotropic kernels to isotropic spacing.
            preprocess (callable or None): Optional preprocessing function for the input.
        
        Returns:
            torch.nn.Module: The instantiated SegResNetDS model on the specified device.
        """
        self.model = SegResNetDS(
            spatial_dims=3,
            init_filters=config['init_filters'],
            in_channels=1,
            out_channels=config['num_classes'],
            act=config['act'],
            norm=config['norm'],
            blocks_down=config['blocks_down'],
            blocks_up=config['blocks_up'],
            dsdepth=config['dsdepth'],
            preprocess=config['preprocess'],
            upsample_mode=config['upsample_mode'],
            resolution=config['resolution']
        )
        return self.model.to(self.device)
    
    def bayesian_search(self, trial):
        """
        Defines the Bayesian optimization search space and builds the model with suggested parameters.
        
        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
        
        Returns:
            torch.nn.Module: The model built with hyperparameters suggested by the trial.
        """
        # Define search space parameters
        init_filters = trial.suggest_categorical("init_filters", [16, 32, 64])
        dsdepth = trial.suggest_int("dsdepth", 1, 3)
        blocks_down = trial.suggest_categorical("blocks_down", [(1, 2, 2, 4), (1, 2, 2, 2), (1, 1, 2, 2)])
        act = trial.suggest_categorical("act", ['relu', 'leaky_relu', "LeakyReLU", "PReLU", "GELU", "ELU"])
        norm = trial.suggest_categorical("norm", ['batch', 'instance'])
        upsample_mode = trial.suggest_categorical("upsample_mode", ['deconv', 'trilinear'])
        
        self.config = {
            'init_filters': init_filters,
            'blocks_down': blocks_down,
            'dsdepth': dsdepth,
            'act': act,
            'norm': norm,
            'blocks_up': None,  # using default upsampling blocks
            'upsample_mode': upsample_mode,
            'resolution': None,
            'preprocess': None
        }
        return self.build_model(self.config)

    def get_model_parameters(self):
        """
        Retrieve stored model parameters.
        
        Returns:
            dict: A dictionary of key model parameters.
        
        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return self.config
