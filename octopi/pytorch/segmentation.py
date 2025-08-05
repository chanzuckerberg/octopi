from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.multiprocessing import Pool
from monai.data import MetaTensor
from monai.transforms import (
    Compose, AsDiscrete, Activations
)
from typing import List, Optional, Union
from octopi.datasets import io as dataio
from copick_utils.io import writers
from octopi.models import common
import torch, copick, gc, os
from octopi.utils import io
from tqdm import tqdm
import numpy as np

class Predictor:

    def __init__(self, 
                 config: str,
                 model_config: Union[str, List[str]],
                 model_weights: Union[str, List[str]],
                 apply_tta: bool = True,
                 device: Optional[str] = None):

        # Open the Copick Project
        self.config = config
        self.root = copick.from_file(config)
        
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available.")

        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print('Running Inference On: ', self.device)

        # Initialize TTA if enabled
        self.apply_tta = apply_tta
        self.create_tta_augmentations() 

        # Determine if Model Soup is Enabled
        if isinstance(model_weights, str):
            model_weights = [model_weights]
        self.apply_modelsoup = len(model_weights) > 1

        # Handle Single Model Config or Multiple Model Configs
        if isinstance(model_config, str):
            model_config = [model_config] * len(model_weights)
        elif len(model_config) != len(model_weights):
            raise ValueError("Number of model configs must match number of model weights.")

        # Load the model(s)
        self._load_models(model_config, model_weights)    

    def _load_models(self, model_config: List[str], model_weights: List[str]):
        """Load a single model or multiple models for soup."""
        
        self.models = []
        for i, (config_path, weights_path) in enumerate(zip(model_config, model_weights)):

            # Load the Model Config and Model Builder
            current_modelconfig = io.load_yaml(config_path)
            model_builder = common.get_model(current_modelconfig['model']['architecture'])

            # Check if the weights file exists
            if not os.path.exists(weights_path):    
                raise ValueError(f"Model weights file does not exist: {weights_path}")
            
            # Create model
            model_builder.build_model(current_modelconfig['model'])
            model = model_builder.model
            
            # Load weights
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
        
        # For backward compatibility, also set self.model to the first model
        self.model = self.models[0]

        # Set the Number of Classes and Input Dimensions - Assume All Models are the Same
        self.Nclass = current_modelconfig['model']['num_classes']     
        self.dim_in = current_modelconfig['model']['dim_in']
        self.input_dim = None

    def _run_single_model_inference(self, model, input_data):
        """Run sliding window inference on a single model."""
        return sliding_window_inference(
            inputs=input_data,
            roi_size=(self.dim_in, self.dim_in, self.dim_in),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )

    def _apply_tta_single_model(self, model, single_sample):
        """Apply TTA to a single model and single sample."""
        # Initialize probability accumulator
        acc_probs = torch.zeros(
            (1, self.Nclass, *single_sample.shape[2:]), 
            dtype=torch.float32, device=self.device
        )
        
        # Process each augmentation
        with torch.no_grad():
            for tta_transform, inverse_transform in zip(self.tta_transforms, self.inverse_tta_transforms):
                # Apply transform
                aug_sample = tta_transform(single_sample)
                
                # Run inference
                predictions = self._run_single_model_inference(model, aug_sample)
                
                # Get softmax probabilities
                probs = torch.softmax(predictions[0], dim=0)
                
                # Apply inverse transform
                inv_probs = inverse_transform(probs)
                
                # Accumulate probabilities
                acc_probs[0] += inv_probs
                
                # Clear memory
                del predictions, probs, inv_probs, aug_sample
                torch.cuda.empty_cache()
        
        # Average accumulated probabilities
        acc_probs = acc_probs / len(self.tta_transforms)
        
        return acc_probs[0]  # Return shape [Nclass, Z, Y, X]

    def _run_inference(self, input_data):
        """
        Main inference function that handles all combinations - Model Soup and/or TTA
        """
        batch_size = input_data.shape[0]
        results = []
        
        # Process one sample at a time for memory efficiency
        for sample_idx in range(batch_size):
            single_sample = input_data[sample_idx:sample_idx+1]
            
            # Initialize probability accumulator for this sample
            acc_probs = torch.zeros(
                (self.Nclass, *single_sample.shape[2:]), 
                dtype=torch.float32, device=self.device
            )
            
            # Process each model
            with torch.no_grad():
                for model in self.models:
                    # Apply TTA with this model
                    if self.apply_tta:
                        model_probs = self._apply_tta_single_model(model, single_sample)
                    # Run inference without TTA
                    else:
                        predictions = self._run_single_model_inference(model, single_sample)
                        model_probs = torch.softmax(predictions[0], dim=0)
                        del predictions
                    
                    # Accumulate probabilities from this model
                    acc_probs += model_probs
                    del model_probs
                    torch.cuda.empty_cache()
            
            # Average probabilities across models (and TTA augmentations if applied)
            acc_probs = acc_probs / len(self.models)
            
            # Convert to discrete prediction
            discrete_pred = torch.argmax(acc_probs, dim=0)
            results.append(discrete_pred)
            
            # Clear memory
            del acc_probs, discrete_pred
            torch.cuda.empty_cache()
        
        return results
        
    def predict_on_gpu(self, 
                        runIDs: List[str],
                        voxel_spacing: float,
                        tomo_algorithm: str):

        # Load data for the current batch
        test_loader, test_dataset = dataio.create_predict_dataloader(
            self.root,
            voxel_spacing, tomo_algorithm,
            runIDs)
        
        # Determine Input Crop Size.
        if self.input_dim is None:
            self.input_dim = dataio.get_input_dimensions(test_dataset, self.dim_in)
        
        predictions = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                tomogram = data['image'].to(self.device)
                data['pred'] = self._run_inference(tomogram)
                
                for idx in range(len(data['image'])):
                    predictions.append(data['pred'][idx].squeeze(0).numpy(force=True)) 

        return predictions

    def batch_predict(self, 
                      num_tomos_per_batch: int = 15, 
                      runIDs: Optional[List[str]] = None,
                      voxel_spacing: float = 10,
                      tomo_algorithm: str = 'denoised', 
                      segmentation_name: str = 'prediction',
                      segmentation_user_id: str = 'octopi',
                      segmentation_session_id: str = '0'):
        """Run inference on tomograms in batches."""                          
        
        # If runIDs are not provided, load all runs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is not None]
            skippedRunIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is None]

            if skippedRunIDs:
                print(f"Warning: skipping runs with no voxel spacing {voxel_spacing}: {skippedRunIDs}")

        # Iterate over batches of runIDs
        for i in range(0, len(runIDs), num_tomos_per_batch):
            # Get a batch of runIDs
            batch_ids = runIDs[i:i + num_tomos_per_batch]  
            print('Running Inference on the Following RunIDs: ', batch_ids)

            predictions = self.predict_on_gpu(batch_ids, voxel_spacing, tomo_algorithm)

            # Save Predictions to Corresponding RunID
            for ind in range(len(batch_ids)):
                run = self.root.get_run(batch_ids[ind])
                seg = predictions[ind]
                writers.segmentation(run, seg, segmentation_user_id, segmentation_name, 
                                   segmentation_session_id, voxel_spacing)

            # After processing and saving predictions for a batch:
            del predictions  # Remove reference to the list holding prediction arrays
            torch.cuda.empty_cache()  # Clear unused GPU memory
            gc.collect()  # Trigger garbage collection for CPU memory

        print('Predictions Complete!')

    def create_tta_augmentations(self):
        """Define TTA augmentations and inverse transforms."""
        # Rotate around the YZ plane (dims 3,4 for input, dims 2,3 for output)
        self.tta_transforms = [
            lambda x: x,                                    # Identity (no augmentation)
            lambda x: torch.rot90(x, k=1, dims=(3, 4)),   # 90° rotation
            lambda x: torch.rot90(x, k=2, dims=(3, 4)),   # 180° rotation
            lambda x: torch.rot90(x, k=3, dims=(3, 4)),   # 270° rotation
        ]

        # Define inverse transformations (flip back to original orientation)
        self.inverse_tta_transforms = [
            lambda x: x,                                    # Identity (no transformation needed)
            lambda x: torch.rot90(x, k=-1, dims=(2, 3)),  # Inverse of 90° (i.e. -90°)
            lambda x: torch.rot90(x, k=-2, dims=(2, 3)),  # Inverse of 180° (i.e. -180°)
            lambda x: torch.rot90(x, k=-3, dims=(2, 3)),  # Inverse of 270° (i.e. -270°)
        ]

###################################################################################################################################################

class MultiGPUPredictor(Predictor):
    
    def __init__(self, 
                 config: str,
                 model_config: Union[str, List[str]],
                 model_weights: Union[str, List[str]],
                 apply_tta: bool = True):
        
        # Initialize parent normally
        super().__init__(config, model_config, model_weights, apply_tta)
        
        self.num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {self.num_gpus}")
        
        # Only create GPU-specific models if we have multiple GPUs
        if self.num_gpus > 1:
            self._create_gpu_models()
    
    def _create_gpu_models(self):
        """Create separate model instances for each GPU."""
        self.gpu_models = {}
        
        for gpu_id in range(self.num_gpus):
            device = torch.device(f'cuda:{gpu_id}')
            gpu_models = []
            
            # Copy each model to this GPU
            for model in self.models:
                gpu_model = type(model)()
                gpu_model.load_state_dict(model.state_dict())
                gpu_model.to(device)
                gpu_model.eval()
                gpu_models.append(gpu_model)
            
            self.gpu_models[gpu_id] = gpu_models
            print(f"Models loaded on GPU {gpu_id}")

    def _run_on_gpu(self, gpu_id: int, batch_ids: List[str], 
                    voxel_spacing: float, tomo_algorithm: str,
                    segmentation_name: str, segmentation_user_id: str, 
                    segmentation_session_id: str):
        """Run inference on a specific GPU for a batch of runs."""
        device = torch.device(f'cuda:{gpu_id}')
        
        # Temporarily switch to this GPU's models and device
        original_device = self.device
        original_models = self.models
        
        self.device = device
        self.models = self.gpu_models[gpu_id]
        
        try:
            print(f"GPU {gpu_id} processing runs: {batch_ids}")
            
            # Run prediction using parent class method
            predictions = self.predict_on_gpu(batch_ids, voxel_spacing, tomo_algorithm)
            
            # Save predictions
            for idx, run_id in enumerate(batch_ids):
                run = self.root.get_run(run_id)
                seg = predictions[idx]
                writers.segmentation(run, seg, segmentation_user_id, 
                                  segmentation_name, segmentation_session_id, 
                                  voxel_spacing)
            
            # Clean up
            del predictions
            torch.cuda.empty_cache()
            
        finally:
            # Restore original settings
            self.device = original_device
            self.models = original_models

    def multigpu_batch_predict(self, 
                              num_tomos_per_batch: int = 15, 
                              runIDs: Optional[List[str]] = None,
                              voxel_spacing: float = 10,
                              tomo_algorithm: str = 'denoised', 
                              segmentation_name: str = 'prediction',
                              segmentation_user_id: str = 'octopi',
                              segmentation_session_id: str = '0'):
        """Run inference across multiple GPUs using threading."""
        
        # Get runIDs if not provided
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is not None]
            skippedRunIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is None]
            if skippedRunIDs:
                print(f"Warning: skipping runs with no voxel spacing {voxel_spacing}: {skippedRunIDs}")

        # Split runIDs into batches
        batches = [runIDs[i:i + num_tomos_per_batch] 
                  for i in range(0, len(runIDs), num_tomos_per_batch)]
        
        print(f"Processing {len(batches)} batches across {self.num_gpus} GPUs")
        
        # Create work queue
        batch_queue = queue.Queue()
        for batch in batches:
            batch_queue.put(batch)
        
        def worker(gpu_id):
            while True:
                try:
                    batch_ids = batch_queue.get_nowait()
                    self._run_on_gpu(gpu_id, batch_ids, voxel_spacing, tomo_algorithm,
                                   segmentation_name, segmentation_user_id, 
                                   segmentation_session_id)
                    batch_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error on GPU {gpu_id}: {e}")
                    batch_queue.task_done()
        
        # Start worker threads for each GPU
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(worker, gpu_id) for gpu_id in range(self.num_gpus)]
            for future in futures:
                future.result()
        
        print('Multi-GPU predictions complete!')

    def batch_predict(self, 
                      num_tomos_per_batch: int = 15, 
                      runIDs: Optional[List[str]] = None,
                      voxel_spacing: float = 10,
                      tomo_algorithm: str = 'denoised', 
                      segmentation_name: str = 'prediction',
                      segmentation_user_id: str = 'octopi',
                      segmentation_session_id: str = '0'):
        """Smart batch predict: uses multi-GPU if available, otherwise single GPU."""
        
        if self.num_gpus > 1:
            print("Using multi-GPU inference")
            self.multigpu_batch_predict(
                num_tomos_per_batch=num_tomos_per_batch,
                runIDs=runIDs,
                voxel_spacing=voxel_spacing,
                tomo_algorithm=tomo_algorithm,
                segmentation_name=segmentation_name,
                segmentation_user_id=segmentation_user_id,
                segmentation_session_id=segmentation_session_id
            )
        else:
            print("Using single GPU inference")
            super().batch_predict(
                num_tomos_per_batch=num_tomos_per_batch,
                runIDs=runIDs,
                voxel_spacing=voxel_spacing,
                tomo_algorithm=tomo_algorithm,
                segmentation_name=segmentation_name,
                segmentation_user_id=segmentation_user_id,
                segmentation_session_id=segmentation_session_id
            )