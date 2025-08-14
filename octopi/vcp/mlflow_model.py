"""
Minimal MLflow wrapper for Octopi that leverages existing octopi inference.
"""

from octopi.models.common import get_model
from octopi.utils.io import load_yaml
import json, torch, yaml, os, tempfile
from typing import Any, Dict
import mlflow.pyfunc
import numpy as np


class OctopiMLflowModel(mlflow.pyfunc.PythonModel):
    """Minimal MLflow wrapper that uses octopi's existing inference capabilities."""
    
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model using octopi's model loading system."""
        
        # Paths to model artifacts
        self.model_config_path = os.path.join(context.artifacts["model_config"], "config.yaml")
        self.model_weights_path = os.path.join(context.artifacts["model_weights"], "best_model.pth")
        
        # Load model config
        self.config = load_yaml(self.model_config_path)
        
        # Initialize model using octopi's model builder
        model_builder = get_model(self.config['model']['architecture'])
        model_builder.build_model(self.config['model'])
        self.model = model_builder.model
        
        # Load weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(self.model_weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Store model params for inference
        self.num_classes = self.config['model']['num_classes']
        self.dim_in = self.config['model']['dim_in']
    
    def _get_input(self, uri: str, **params) -> Any:
        """
        Load input tomogram file.
        
        Args:
            uri: Path to copick project config or tomogram file
            **params: Additional parameters
            
        Returns:
            Input data for octopi inference
        """
        # For now, just return the URI - octopi's existing data loaders handle file loading
        return uri
    
    def _forward(self, input_obj: Any, **params) -> np.ndarray:
        """
        Run inference using octopi's prediction pipeline.
        
        Args:
            input_obj: Input data (file path or copick config)
            **params: Additional parameters (voxel_spacing, tomo_algorithm, etc.)
            
        Returns:
            Segmentation predictions
        """
        # Use octopi's existing Predictor class for inference
        from octopi.pytorch.segmentation import Predictor
        
        # Create temporary config for predictor
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary model config file
            temp_config = os.path.join(temp_dir, "temp_config.yaml")
            with open(temp_config, 'w') as f:
                yaml.dump(self.config, f)
            
            # Create predictor with current model
            predictor = Predictor(
                config=input_obj,  # Copick project config
                model_config=temp_config,
                model_weights=self.model_weights_path,
                apply_tta=params.get('apply_tta', False),
                device=str(self.device)
            )
            
            # Run prediction
            voxel_spacing = params.get('voxel_spacing', 10)
            tomo_algorithm = params.get('tomo_algorithm', 'denoised')
            run_ids = params.get('run_ids', None)
            
            predictions = predictor.predict_on_gpu(
                runIDs=run_ids,
                voxel_spacing=voxel_spacing,
                tomo_algorithm=tomo_algorithm
            )
            
            return predictions[0] if predictions else np.array([])
    
    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input, params=None):
        """
        Main prediction method.
        
        Args:
            context: MLflow context
            model_input: DataFrame with 'input_uri' column (copick config paths)
            params: Optional parameters
            
        Returns:
            List of segmentation predictions
        """
        if params is None:
            params = {}
            
        predictions = []
        
        # Handle DataFrame input
        if hasattr(model_input, 'iterrows'):
            for _, row in model_input.iterrows():
                uri = row['input_uri']
                # Run prediction using octopi's existing pipeline
                prediction = self._forward(uri, **params)
                predictions.append(prediction)
        else:
            # Handle single input
            prediction = self._forward(model_input, **params)
            predictions.append(prediction)
        
        return predictions
