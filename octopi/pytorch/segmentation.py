from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.multiprocessing import Pool
from monai.data import MetaTensor
from monai.transforms import (
    Compose, AsDiscrete, Activations
)
import octopi.processing.writers as write
from octopi.models import common
from typing import List, Optional
import torch, copick, gc, os
from octopi import io, utils
from tqdm import tqdm
import numpy as np

class Predictor:

    def __init__(self, 
                 config: str,
                 model_config: str,
                 model_weights: str,
                 apply_tta: bool = True,
                 device: Optional[str] = None):

        self.config = config
        self.root = copick.from_file(config)

        # Load the model config
        model_config = utils.load_yaml(model_config)

        self.Nclass = model_config['model']['num_classes']     
        self.dim_in = model_config['model']['dim_in']
        self.input_dim = None
        
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

        # Check to see if the model weights file exists
        if not os.path.exists(model_weights):
            raise ValueError(f"Model weights file does not exist: {model_weights}")

        # Load the model weights
        model_builder = common.get_model(model_config['model']['architecture'])
        model_builder.build_model(model_config['model'])
        self.model = model_builder.model
        state_dict = torch.load(model_weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

       # Initialize TTA if enabled
        self.apply_tta = apply_tta
        if self.apply_tta: 
            self.create_tta_augmentations() 
            # self.post_transforms = Compose([
            #     Activations(softmax=True)  # Keep probability output
            # ])
            self.softmax_transform = Compose([
                Activations(softmax=True)  # Keep probability output
            ])
            
            # Create the final discretization transform
            self.discretize_transform = AsDiscrete(argmax=True)
        else:
            # Define the post-processing transforms
            self.post_transforms = Compose([
                Activations(softmax=True),
                AsDiscrete(argmax=True)
            ])
            
        
    def _run_inference(self, input):
        """Apply sliding window inference to the input."""
        with torch.no_grad():
            predictions = sliding_window_inference(
                inputs=input,
                roi_size=(self.dim_in, self.dim_in, self.dim_in),
                sw_batch_size=4,  # one window is proecessed at a time
                predictor=self.model,
                overlap=0.5,
            )
            return [self.post_transforms(i) for i in decollate_batch(predictions)]

    def _run_inference_tta(self, input_data):
        """Memory-efficient TTA implementation that returns proper discrete segmentation maps."""
        
        batch_size = input_data.shape[0]
        results = []
        
        # Process one sample at a time
        for sample_idx in range(batch_size):
            # Extract single sample
            single_sample = input_data[sample_idx:sample_idx+1]
            
            # Initialize probability accumulator for this sample
            # Shape: [1, Nclass, Z, Y, X]
            acc_probs = torch.zeros(
                (1, self.Nclass, *single_sample.shape[2:]), 
                dtype=torch.float32, device=self.device
            )
            
            # Process each augmentation
            with torch.no_grad():
                for tta_transform, inverse_transform in zip(self.tta_transforms, self.inverse_tta_transforms):
                    # Apply transform to single sample
                    aug_sample = tta_transform(single_sample)
                    
                    # Free memory
                    torch.cuda.empty_cache()
                    
                    # Run inference (one sample at a time)
                    predictions = sliding_window_inference(
                        inputs=aug_sample,
                        roi_size=(self.dim_in, self.dim_in, self.dim_in),
                        sw_batch_size=4,  # Process one window at a time
                        predictor=self.model,
                        overlap=0.5,
                    )
                    
                    # Get softmax probabilities
                    probs = self.softmax_transform(predictions[0])  # Get first (only) item
                    
                    # Apply inverse transform with correct dimensions
                    inv_probs = inverse_transform(probs)
                    
                    # Accumulate probabilities
                    acc_probs[0] += inv_probs
                    
                    # Clear memory
                    del predictions, probs, inv_probs, aug_sample
            
            # Average accumulated probabilities
            acc_probs = acc_probs / len(self.tta_transforms)
            
            # Convert to discrete prediction - get argmax along class dimension
            # This gives us a tensor of shape [1, Z, Y, X] with discrete class indices
            discrete_pred = torch.argmax(acc_probs, dim=1)
            
            # Add to results - keeping only the spatial dimensions [Z, Y, X]
            results.append(discrete_pred[0])
            
            # Clear memory
            del acc_probs, discrete_pred
            torch.cuda.empty_cache()
        
        return results
        
    def predict_on_gpu(self, 
                        runIDs: List[str],
                        voxel_spacing: float,
                        tomo_algorithm: str ):

        # Load data for the current batch
        test_loader, test_dataset = io.create_predict_dataloader(
            self.root,
            voxel_spacing, tomo_algorithm,
            runIDs)
        
        # Determine Input Crop Size.
        if self.input_dim is None:
            self.input_dim = io.get_input_dimensions(test_dataset, self.dim_in)
        
        predictions = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                tomogram = data['image'].to(self.device)
                if self.apply_tta: data['pred'] = self._run_inference_tta(tomogram)
                else:              data['pred']  = self._run_inference(tomogram)
                for idx in range(len(data['image'])):
                    predictions.append(data['pred'][idx].squeeze(0).numpy(force=True)) 

        return predictions

    def batch_predict(self, 
                      num_tomos_per_batch = 15, 
                      runIDs: Optional[str] = None,
                      voxel_spacing: float = 10,
                      tomo_algorithm: str = 'denoised', 
                      segmentation_name: str = 'prediction',
                      segmentation_user_id: str = 'octopi',
                      segmentation_session_id: str = '0'):

        """Run inference on tomograms in batches."""                          
        
        # If runIDs are not provided, load all runs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs]
        
        # Iterate over batches of runIDs
        for i in range(0, len(runIDs), num_tomos_per_batch):

            # Get a batch of runIDs
            batch_ids = runIDs[i:i + num_tomos_per_batch]  
            print('Running Inference on the Follow RunIDs: ', batch_ids)

            predictions = self.predict_on_gpu(batch_ids, voxel_spacing, tomo_algorithm)

            # Save Predictions to Corresponding RunID
            for ind in range(len(batch_ids)):
                run = self.root.get_run(batch_ids[ind])
                seg = predictions[ind]
                write.segmentation(run, seg, segmentation_user_id, segmentation_name, 
                                   segmentation_session_id, voxel_spacing)

            # After processing and saving predictions for a batch:
            del predictions  # Remove reference to the list holding prediction arrays
            torch.cuda.empty_cache()  # Clear unused GPU memory
            gc.collect()  # Trigger garbage collection for CPU memory

        print('Predictions Complete!')

    def create_tta_augmentations(self):
        """Define TTA augmentations and inverse transforms."""

        # Instead of Flip lets rotate around the first axis 3 times (90,180,270)
        self.tta_transforms = [
            lambda x: x,                    # Identity (no augmentation)
            lambda x: torch.rot90(x, k=1, dims=(3, 4)),  # 90° rotation
            lambda x: torch.rot90(x, k=2, dims=(3, 4)),  # 180° rotation
            lambda x: torch.rot90(x, k=3, dims=(3, 4)),  # 270° rotation
            # Flip(spatial_axis=0),         # Flip along x-axis (depth)
            # Flip(spatial_axis=1),         # Flip along y-axis (height)
            # Flip(spatial_axis=2),         # Flip along z-axis (width)
        ]

        # Define inverse transformations (flip back to original orientation)
        self.inverse_tta_transforms = [
            lambda x: x,                           # Identity (no transformation needed)
            lambda x: torch.rot90(x, k=-1, dims=(2, 3)),  # Inverse of 90° (i.e. -90°)
            lambda x: torch.rot90(x, k=-2, dims=(2, 3)),  # Inverse of 180° (i.e. -180°)
            lambda x: torch.rot90(x, k=-3, dims=(2, 3)),  # Inverse of 270° (i.e. -270°)
            # Flip(spatial_axis=0),         # Undo Flip along x-axis
            # Flip(spatial_axis=1),         # Undo Flip along y-axis
            # Flip(spatial_axis=2),         # Undo Flip along z-axis
        ]

###################################################################################################################################################

class MultiGPUPredictor(Predictor):

    def __init__(self, 
                 config: str,
                 model_config: str,
                 model_weights: str):
        super().__init__(config, model_config, model_weights)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise RuntimeError("MultiGPUPredictor requires at least 2 GPUs.")
    
    def predict_on_gpu(self, gpu_id: int, batch_ids: List[str], voxel_spacing: float, tomo_algorithm: str) -> List[np.ndarray]:
        """Helper function to run inference on a single GPU."""
        device = torch.device(f'cuda:{gpu_id}')
        self.model.to(device)
        
        # Load data specific to the batch assigned to this GPU
        test_loader = io.load_predict_data(self.root, batch_ids, voxel_spacing, tomo_algorithm)
        predictions = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"GPU {gpu_id}"):
                tomogram = data['image'].to(device)
                data["prediction"] = self.run_inference(tomogram)
                data = [self.post_processing(i) for i in decollate_batch(data)]
                for b in data:
                    predictions.append(b['prediction'].squeeze(0).cpu().numpy())
                    
        return predictions

    def multi_gpu_inference(self, 
                            num_tomos_per_batch: int = 15, 
                            runIDs: Optional[List[str]] = None,
                            voxel_spacing: float = 10,
                            tomo_algorithm: str = 'denoised', 
                            save: bool = False,
                            segmentation_name: str = 'prediction',
                            segmentation_user_id: str = 'monai',
                            segmentation_session_id: str = '0') -> Optional[List[np.ndarray]]:
        """Run inference across multiple GPUs, optionally saving results or returning predictions."""
        
        runIDs = runIDs or [run.name for run in self.root.runs]
        all_predictions = []

        # Divide runIDs into batches for each GPU
        batches = [runIDs[i:i + num_tomos_per_batch] for i in range(0, len(run_ids), num_tomos_per_batch)]
        
        # Run inference in parallel across GPUs
        for i in range(0, len(batches), self.num_gpus):
            gpu_batches = batches[i:i + self.num_gpus]
            with Pool(processes=self.num_gpus) as pool:
                results = pool.starmap(
                    self.predict_on_gpu,
                    [(gpu_id, gpu_batches[gpu_id], voxel_spacing, tomo_algorithm) for gpu_id in range(len(gpu_batches))]
                )

            # Collect and save results
            for gpu_id, predictions in enumerate(results):
                if save:
                    for idx, run_id in enumerate(gpu_batches[gpu_id]):
                        run = self.root.get_run(run_id)
                        segmentation = predictions[idx]
                        write.segmentation(run, segmentation, segmentation_user_id, segmentation_name, 
                                           segmentation_session_id, voxel_spacing)
                else:
                    all_predictions.extend(predictions)

        print('Multi-GPU predictions complete.')
        
        return None if save else all_predictions