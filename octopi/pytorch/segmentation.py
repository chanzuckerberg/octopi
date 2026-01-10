from __future__ import annotations
from monai.inferers import sliding_window_inference
from octopi.utils.progress import _progress
from typing import List, Optional, Union
from octopi.datasets import io as dataio
import torch, copick, gc, os, pprint
from copick_utils.io import writers
import torch.multiprocessing as mp
from dataclasses import dataclass
from octopi.models import common
from octopi.utils import io
from tqdm import tqdm
import numpy as np

from monai.transforms import (
    Compose, 
    NormalizeIntensity,
    EnsureChannelFirst,  
)

# Single GPU Inference Support
class Predictor:
    def __init__(self, 
                 config: str,
                 model_config: Union[str, List[str]],
                 model_weights: Union[str, List[str]],
                 apply_tta: bool = True,
                 device: Optional[str] = None,
                 rank: int = 0
        ):
        """
        Predictor class for single GPU inference with optional test-time augmentation (TTA) 

        Args:
            config (str): Path to the Copick project configuration file.
            model_config (Union[str, List[str]]): Path(s) to the model configuration file(s).
            model_weights (Union[str, List[str]]): Path(s) to the model weight file(s).
            apply_tta (bool, optional): Whether to apply test-time augmentation. Defaults to True.
            device (Optional[str], optional): Device to run inference on. Defaults to None, which selects 'cuda' if available.
            rank (int, optional): Rank of the current process for distributed inference. Defaults to 0.
        """

        # Open the Copick Project
        self.config = config
        self.root = copick.from_file(config)
        
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available.")

        # Initialize TTA if enabled
        self.apply_tta = apply_tta
        self.create_tta_augmentations() 

        # Determine if Model Soup is Enabled
        if isinstance(model_weights, str):
            model_weights = [model_weights]
        self.apply_modelsoup = len(model_weights) > 1
        self.model_weights = model_weights

        # Sliding Window Inference Parameters
        self.sw_bs = 4 # sliding window batch size
        self.overlap = 0.5 # overlap between windows
        self.sw = None

        # Set the device
        self.device = torch.device('cuda') if device else device    

        # Handle Single Model Config or Multiple Model Configs
        if isinstance(model_config, str):
            model_config = [model_config] * len(model_weights)
        elif len(model_config) != len(model_weights):
            raise ValueError("Number of model configs must match number of model weights.")
        self.model_config = model_config            

        # Load the model(s)
        self._load_models(model_config, model_weights)

        # Diagnostics 
        self.rank = rank

    def _print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)        

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

        # Print a message if Model Soup is Enabled
        if self.apply_modelsoup:
            self._print(f'Model Soup is Enabled : {len(self.models)} models loaded for ensemble inference')
    
    @torch.inference_mode()
    def _run_single_model_inference(self, model, input_data):
        """Run sliding window inference on a single model."""
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            return sliding_window_inference(
                inputs=input_data,
                roi_size=(self.dim_in, self.dim_in, self.dim_in),
                sw_batch_size=self.sw_bs,
                predictor=model,
                overlap=self.overlap,
            )

    @torch.inference_mode()
    def _apply_tta_single_model(self, model, single_sample):
        """Apply TTA to a single model and single sample."""
        # Initialize probability accumulator
        acc_probs = torch.zeros(
            (1, self.Nclass, *single_sample.shape[2:]), 
            dtype=torch.float32, device=self.device
        )
        
        # Process each augmentation
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
        # Overwrite sw_bs with sw if provided
        if self.sw is not None:
            self.sw_bs = self.sw
        
        # Get the batch size (# of tomograms)
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

    def predict(self, input_data):
        """Run Prediction from an Input Tomogram.
        Args:
            input_data (torch.Tensor or np.ndarray): Input tomogram of shape [Z, Y, X]
        Returns:
            Predicted segmentation mask of shape [Z, Y, X]
        """
        
        is_numpy = False
        if isinstance(input_data, np.ndarray):
            is_numpy = True
            input_data = torch.from_numpy(input_data)
        
        # Apply transforms directly to tensor (no dictionary needed)
        pre_transforms = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            NormalizeIntensity(),                         
        ])
        
        input_data = pre_transforms(input_data)
        
        # Add batch dimension and move to device
        input_data = input_data.unsqueeze(0).to(self.device)
        
        # Run inference
        pred = self._run_inference(input_data)[0]
        
        if is_numpy:
            pred = pred.cpu().numpy()
        return pred

    def batch_predict(
        self, 
        num_tomos_per_batch: int = 4, 
        runIDs: Optional[List[str]] = None,
        voxel_spacing: float = 10,
        tomo_algorithm: str = 'denoised', 
        name: str = 'predict',
        userid: str = 'octopi',
        sessionid: str = '1',
        progress_q = None ):
        """
        Run inference on tomograms in batches.
        
        Parameters:
        - num_tomos_per_batch: Number of tomograms to process per batch.
        - runIDs: List of run IDs to process. If None, all runs are processed.
        - voxel_spacing: Voxel spacing for the tomograms.
        - tomo_algorithm: Tomography algorithm to use.
        - name: Name of the segmentation.
        - userid: User ID for the segmentation.
        - sessionid: Session ID for the segmentation.
        """
        
        # If runIDs are not provided, load all runs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is not None]
            skippedRunIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is None]

            if skippedRunIDs:
                self._print(f"Warning: skipping runs with no voxel spacing {voxel_spacing}: {skippedRunIDs}")

        # Create Dataloader for the Copick Config and RunIDs
        self.test_loader, self.test_dataset = dataio.create_predict_dataloader(
            self.config, voxel_spacing, tomo_algorithm, runIDs, num_tomos_per_batch
        )

        # Save the inference parameters
        seg_info = [name, userid, sessionid]
        if self.rank == 0: self.save_parameters(tomo_algorithm, voxel_spacing, seg_info)

        # Determine Input Crop Size.
        if self.input_dim is None:
            self.input_dim = dataio.get_input_dimensions(self.test_dataset, self.dim_in)
        
        # Main Loop for Inference
        for data in _progress(self.test_loader, description=f"Segmenting Tomograms on GPU: {self.device}"):
            
            # Run inference on the tomogram
            tomogram = data['image'].to(self.device)
            preds = self._run_inference(tomogram)
            
            # Write the Prediction to the corresponding runs
            for i in range(len(preds)):
                pred = preds[i].cpu().numpy()
                run = self.root.get_run(data['runid'][i])
                writers.segmentation(
                    run, pred,
                    userid, name, sessionid,
                    voxel_spacing
                )

        print('✅ Predictions Complete!')

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

    def save_parameters(self, 
        tomo_algorithm: str, 
        voxel_size: float, 
        seg_info: List[str]
    ):
        """
        Save inference parameters to a YAML file for record-keeping and reproducibility.
        """

        # Load the model config
        model_config = io.load_yaml(self.model_config[0])

        # Create parameters dictionary
        params = {
            "inputs": {
                "config": self.config,
                "tomo_alg": tomo_algorithm,
                "voxel_size": voxel_size
            },
            'model': {
                'configs': self.model_config,
                'weights': self.model_weights
            },
            'labels': model_config['labels'],
            "outputs": {
                "seg_name": seg_info[0],
                "seg_user_id": seg_info[1],
                "seg_session_id": seg_info[2]
            }
        }

        # Print the parameters
        print(f"\nParameters for Inference (Segment Prediction):")
        pprint.pprint(params); print()

        # Save to YAML file
        overlay_root = io.remove_prefix(self.root.config.overlay_root)
        basepath = os.path.join(overlay_root, 'logs')
        os.makedirs(basepath, exist_ok=True)
        output_path = os.path.join(
            basepath,
            f'segment-{seg_info[1]}_{seg_info[2]}_{seg_info[0]}.yaml')
        io.save_parameters_yaml(params, output_path)

#########################################################################################################
# Multi-GPU Inference Support

@dataclass
class _JobSpec:
    config: str
    model_config: Union[str, List[str]]
    model_weights: Union[str, List[str]]
    apply_tta: bool

    runIDs: List[str]
    num_tomos_per_batch: int
    tomo_algorithm: str
    voxel_spacing: float
    name: str
    userid: str
    sessionid: str


def _shard_round_robin(runIDs: List[str], n: int) -> List[List[str]]:
    """Evenly distribute runIDs across n shards."""
    shards = [[] for _ in range(n)]
    for i, rid in enumerate(runIDs):
        shards[i % n].append(rid)
    return shards


def _worker_process(
    local_rank: int,
    job: _JobSpec,
):
    """
    One process per GPU.
    local_rank is 0..(world_size-1) and we map it to cuda:{local_rank}.
    """
    # IMPORTANT for multi-process CUDA
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Optional: reduce CPU thread oversubscription in each process
    # especially on HPC nodes.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count())))

    predictor = Predictor(
        config=job.config,
        model_config=job.model_config,
        model_weights=job.model_weights,
        apply_tta=job.apply_tta,
        device=device,
        rank=local_rank
    )

    # Run inference on this shard only
    if job.runIDs:
        predictor.batch_predict(
            num_tomos_per_batch=job.num_tomos_per_batch,
            runIDs=job.runIDs,
            voxel_spacing=job.voxel_spacing,
            tomo_algorithm=job.tomo_algorithm,
            name=job.name,
            userid=job.userid,
            sessionid=job.sessionid,
        )

# mp.spawn passes (rank, *args). We need job per rank.
# Easiest: pass list and index by rank.
def _spawn_entry(rank: int, jobs_list: List[_JobSpec]):
    _worker_process(rank, jobs_list[rank])

class MultiGpuPredictor:
    """
    Wrapper that mirrors Predictor's batch_predict API, but distributes work across GPUs
    using one process per GPU.
    """

    def __init__(
        self,
        config: str,
        model_config: Union[str, List[str]],
        model_weights: Union[str, List[str]],
        apply_tta: bool = True,
        device: Optional[str] = None,  # ignored; we choose per-process cuda device
    ):
        self.config = config
        self.root = copick.from_file(config)
        self.model_config = model_config
        self.model_weights = model_weights
        self.apply_tta = apply_tta

    def batch_predict(
        self,
        runIDs: List[str],
        num_tomos_per_batch: int = 1,
        tomo_algorithm: str = "denoised",
        voxel_spacing: float = 10.0,
        name: str = "predict",
        userid: str = "octopi",
        sessionid: str = "1",
    ):
        # If only one GPU, just run single process path.
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA GPUs available.")
            
        # Single GPU fallback: just instantiate and run normally
        if world_size == 1:
            predictor = Predictor(
                config=self.config,
                model_config=self.model_config,
                model_weights=self.model_weights,
                apply_tta=self.apply_tta,
                device=torch.device("cuda:0"),
            )
            predictor.batch_predict(
                num_tomos_per_batch=num_tomos_per_batch,
                runIDs=runIDs,
                voxel_spacing=voxel_spacing,
                tomo_algorithm=tomo_algorithm,
                name=name,
                userid=userid,
                sessionid=sessionid,
            )

        # If runIDs are not provided, load all runs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is not None]
            skippedRunIDs = [run.name for run in self.root.runs if run.get_voxel_spacing(voxel_spacing) is None]

            if skippedRunIDs:
                print(f"Warning: skipping runs with no voxel spacing {voxel_spacing}: {skippedRunIDs}")            
                
        # Each process gets the same job spec except runIDs shard differs.
        jobs = []
        shards = _shard_round_robin(runIDs, world_size)
        for shard in shards:
            jobs.append(
                _JobSpec(
                    config=self.config,
                    model_config=self.model_config,
                    model_weights=self.model_weights,
                    apply_tta=self.apply_tta,
                    runIDs=shard,
                    num_tomos_per_batch=num_tomos_per_batch,
                    tomo_algorithm=tomo_algorithm,
                    voxel_spacing=voxel_spacing,
                    name=name,
                    userid=userid,
                    sessionid=sessionid,
                )
            )

        # IMPORTANT: use spawn (not fork) for CUDA safety
        mp.spawn(_spawn_entry, args=(jobs,), nprocs=world_size, join=True)

        print('✅ Segmentations Complete!')