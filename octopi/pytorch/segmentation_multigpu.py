from concurrent.futures import ThreadPoolExecutor
from octopi.pytorch.segmentation import Predictor
from typing import List, Union, Optional
from copick_utils.io import writers
import queue, torch

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