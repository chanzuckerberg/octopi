# Adding New Models

This page covers how to integrate custom neural network architectures into octopi for both direct training and automated model exploration.

## Overview

Octopi supports custom model architectures through a standardized template system. By following the model template structure, you can seamlessly integrate any PyTorch-based architecture into octopi's training and automated hyperparameter search workflows.

Custom models enable you to:

- **Experiment with novel architectures** tailored to your specific particle types
- **Integrate state-of-the-art models** from recent research papers
- **Leverage automated hyperparameter search** for your custom architectures
- **Maintain compatibility** with octopi's training and inference pipelines

## Model Template Structure

All custom models must inherit from the standardized template located at `octopi/models/ModelTemplate.py`. The template defines three essential methods that enable full integration with octopi's workflows.

### Required Methods

- **`build_model(config)`**: Constructs the model from a configuration dictionary
- **`bayesian_search(trial, num_classes)`**: Defines the hyperparameter search space for automated optimization
- **`get_model_parameters()`**: Returns the stored model configuration

### Basic Template

```python
import torch.nn as nn
import torch

class MyCustomModel:
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
        
        Args:
            config (dict): Configuration dictionary containing model parameters
            
        Returns:
            torch.nn.Module: The constructed model
        """
        self.config = config
        
        # Your model construction logic here
        # self.model = YourModelClass(**config)
        
        return self.model

    def bayesian_search(self, trial, num_classes: int):
        """
        Define the hyperparameter search space for Bayesian optimization.
        
        Args:
            trial (optuna.trial.Trial): Optuna trial object for suggesting parameters
            num_classes (int): Number of classes in the dataset
            
        Returns:
            torch.nn.Module: The constructed model with suggested parameters
        """
        # Define search space using trial.suggest_* methods
        # param1 = trial.suggest_int("param1", min_val, max_val)
        # param2 = trial.suggest_categorical("param2", [choice1, choice2])
        
        # Create config dictionary
        self.config = {
            'architecture': 'MyCustomModel',
            'num_classes': num_classes,
            # Add your suggested parameters here
        }
        
        return self.build_model(self.config)
    
    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return self.config
```

## Complete Example: Custom U-Net

Here's a complete example showing how to integrate a custom U-Net architecture:

```python
from monai.networks.nets import UNet
import torch.nn as nn
import torch

class MyUNet:
    def __init__(self):
        # Placeholder for the model and config
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """Creates the U-Net model based on provided parameters."""
        
        self.config = config
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=config['num_classes'],
            channels=config['channels'],
            strides=config['strides'],
            num_res_units=config['num_res_units'],
            dropout=config['dropout']
        )
        return self.model
    
    def bayesian_search(self, trial, num_classes: int):
        """Defines the Bayesian optimization search space and builds the model."""
        
        # Define the search space
        num_layers = trial.suggest_int("num_layers", 3, 5)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        base_channel = trial.suggest_categorical("base_channel", [8, 16, 32])
        num_res_units = trial.suggest_int("num_res_units", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        
        # Create channel sizes and strides
        downsampling_channels = [base_channel * (2 ** i) for i in range(num_layers)]
        hidden_channels = [downsampling_channels[-1]] * hidden_layers
        channels = downsampling_channels + hidden_channels
        strides = [2] * (num_layers - 1) + [1] * hidden_layers

        # Create config dictionary
        self.config = {
            'architecture': 'MyUNet',
            'num_classes': num_classes,
            'channels': channels,
            'strides': strides,
            'num_res_units': num_res_units,
            'dropout': dropout
        }
        
        return self.build_model(self.config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return self.config
```

## Integration Steps

### Step 1: Create Your Model Class

Create a new Python file in the `octopi/models/` directory (e.g., `MyCustomModel.py`) following the template structure shown above.

### Step 2: Register Your Model

Add your model to octopi's model registry by updating the model builder system. This typically involves:

```python
# In your model file or a registry file
from octopi.models.common import register_model

@register_model("MyCustomModel")
class MyCustomModel:
    # Your model implementation
    pass
```

### Step 3: Use in Training

Once registered, your model can be used in training workflows:

```python
from octopi.workflows import train

# Model configuration for your custom architecture
model_config = {
    'architecture': 'MyCustomModel',
    'num_classes': 6,
    'your_param1': value1,
    'your_param2': value2,
    # Add your specific parameters
}

# Train using your custom model
train(
    config=config,
    target_info=target_info,
    tomo_algorithm='denoised',
    voxel_size=10.0,
    loss_function=loss_function,
    model_config=model_config,
    # ... other parameters
)
```

### Step 4: Use in Model Exploration

Your custom model automatically works with automated hyperparameter search:

```python
from octopi.pytorch.model_search_submitter import ModelSearchSubmit

optimizer = ModelSearchSubmit(
    copick_config=config,
    target_name=target_name,
    target_user_id=target_user_id,
    target_session_id=target_session_id,
    tomo_algorithm='denoised',
    voxel_size=10.0,
    Nclass=6,
    model_type='MyCustomModel',  # Use your custom model
    num_trials=50
)

optimizer.run_model_search()
```