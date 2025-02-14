from monai.networks.nets import UNet
import torch.nn as nn
import torch

class ModelTemplate:
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        
        # Placeholder for the model
        self.model = None

    def build_model(self):
        pass

    def bayesian_search(self, trial):
        pass
    
    def get_model_parameters(self):
        pass