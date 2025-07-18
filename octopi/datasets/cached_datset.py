from typing import List, Tuple, Callable, Optional, Dict, Any
from monai.transforms import Compose
from monai.data import CacheDataset
from octopi.datasets import io
from tqdm import tqdm
import os, sys

class MultiConfigCacheDataset(CacheDataset):
    """
    A custom CacheDataset that loads data lazily from multiple sources
    with consolidated loading and caching process.
    """
    
    def __init__(
        self,
        manager,
        run_ids: List[Tuple[str, str]],
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        progress: bool = True,
        copy_cache: bool = True,
        cache_num: int = sys.maxsize
    ):
        # Save reference to manager and run_ids
        self.manager = manager
        self.run_ids = run_ids
        self.progress = progress
        
        # Prepare empty data list first - don't load immediately
        self.data = []
        
        # Initialize the parent CacheDataset with an empty list
        # We'll override the _fill_cache method to handle loading and caching in one step
        super().__init__(
            data=[],  # Empty list - we'll load data in _fill_cache
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=False,  # We'll handle our own progress
            copy_cache=copy_cache,
            cache_num=cache_num
        )
    
    def _fill_cache(self):
        """
        Override the parent's _fill_cache method to combine loading and caching.
        """
        if self.progress:
            print("Loading and caching dataset...")
        
        # Load and process data in a single operation
        self.data = []
        iterator = tqdm(self.run_ids, desc="Loading dataset") if self.progress else self.run_ids
        
        for session_name, run_name in iterator:
            root = self.manager.roots[session_name]
            batch_data = io.load_training_data(
                root, 
                [run_name], 
                self.manager.voxel_size, 
                self.manager.tomo_algorithm, 
                self.manager.target_name, 
                self.manager.target_session_id, 
                self.manager.target_user_id, 
                progress_update=False
            )
            
            self.data.extend(batch_data)
            
            # Process and cache this batch right away
            for i, item in enumerate(batch_data):
                if len(self._cache) < self.cache_num and self.cache_rate > 0.0:
                    if np.random.random() < self.cache_rate:
                        self._cache.append(self._transform(item))
        
        # Check max label value if needed
        if hasattr(self.manager, '_check_max_label_value'):
            self.manager._check_max_label_value(self.data)
        
        # Update the _data attribute to match the loaded data
        self._data = self.data
    
    def __len__(self):
        """
        Return the length of the dataset.
        """
        if not self.data:
            self._fill_cache()  # Load data if not loaded yet
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Return the item at the given index.
        """
        if not self.data:
            self._fill_cache()  # Load data if not loaded yet
        
        # Use parent's logic for cached items
        if index < len(self._cache):
            return self._cache[index]
        
        # Otherwise transform on-the-fly
        return self._transform(self.data[index])

# TODO: Implement Single Config Cache Dataset
# class SingleConfigCacheDataset(CacheDataset):
#     def __init__(self, 
#                  root: Any, 
#                  run_ids: List[str], 
#                  voxel_size: float, 
#                  tomo_algorithm: str, 
#                  target_name: str, 