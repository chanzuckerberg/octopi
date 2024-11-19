from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, 
    RandFlipd, 
    Orientationd, 
    RandRotate90d, 
    NormalizeIntensityd,
    EnsureChannelFirstd, 
    RandCropByLabelClassesd,
    ScaleIntensityRanged,  
)
from model_explore.pytorch.datasets import dataset
from model_explore.pytorch import io
from typing import List, Optional
import torch, os, random

class TrainLoaderManager:

    def __init__(self, 
                 config: str, 
                 target_name: str,
                 target_session_id: str = None,
                 target_user_id: str = None,
                 voxel_size: float = 10, 
                 tomo_algorithm: str = 'wbp', 
                 tomo_batch_size: int = 15, # Number of Tomograms to Load Per Sub-Epoch    
                 Nclasses: int = 3): # Number of Objects + Background

        # Read Copick Projectdd
        self.copick_config = config
        self.root = io.load_copick_config(config)

        # Copick Query for Target
        self.target_name = target_name 
        self.target_session_id = target_session_id
        self.target_user_id = target_user_id

        # Copick Query For Input Tomogram
        self.voxel_size = voxel_size
        self.tomo_algorithm = tomo_algorithm

        self.Nclasses = Nclasses
        self.tomo_batch_size = tomo_batch_size

        self.reload_training_dataset = True
        self.reload_validation_dataset = True
        self.val_loader = None
        self.train_loader = None

    def get_data_splits(self,
                        trainRunIDs: str = None,
                        validateRunIDs: str = None,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        create_test_dataset: bool = True):

        # Option 1: Only TrainRunIDs are Provided, Split into Train, Validate and Test (Optional)
        if trainRunIDs is not None and validateRunIDs is None:
            trainRunIDs, validateRunIDs, testRunIDs = io.split_datasets(trainRunIDs, train_ratio, val_ratio, 
                                                                        test_ratio, create_test_dataset)
        # Option 2: TrainRunIDs and ValidateRunIDs are Provided, No Need to Split
        elif trainRunIDs is not None and validateRunIDs is not None:
            testRunIDs = None
        # Option 3: Use the Entire Copick Project, Split into Train, Validate and Test
        else:
            runIDs = [run.name for run in self.root.runs]
            trainRunIDs, validateRunIDs, testRunIDs = io.split_datasets(runIDs, train_ratio, val_ratio, 
                                                                        test_ratio, create_test_dataset)

        # Swap if Test Runs is Larger than Validation Runs
        if len(testRunIDs) > len(validateRunIDs):
            testRunIDs, validateRunIDs = validateRunIDs, testRunIDs

        # Determine if All Tomograms Fit in Memory, Or if Future Tomograms will
        # Need to be Loaded
        if len(validateRunIDs) < self.tomo_batch_size:
            self.reload_validation_dataset  = False

        if len(trainRunIDs) < self.tomo_batch_size:
            self.reload_training_dataset = False

        # Save the Runs for DataSplits into a Dictionary    
        self.myRunIDs = {}
        self.myRunIDs['train'] = trainRunIDs
        self.myRunIDs['validate'] = validateRunIDs
        self.myRunIDs['test'] = testRunIDs

        print(f"Number of training samples: {len(trainRunIDs)}")
        print(f"Number of validation samples: {len(validateRunIDs)}")
        print(f'Number of test samples: {len(testRunIDs)}')    

        # Define separate batch sizes
        self.train_batch_size = self.tomo_batch_size
        self.val_batch_size = min( len(self.myRunIDs['validate']), self.tomo_batch_size)

        self._initialize_val_iterators()
        self._initialize_train_iterators()

        return self.myRunIDs
    
    def _get_padded_list(self, data_list, batch_size):
        # Calculate padding needed to make `data_list` a multiple of `batch_size`
        remainder = len(data_list) % batch_size
        if remainder > 0:
            # Number of additional items needed to make the length a multiple of batch size
            padding_needed = batch_size - remainder
            # Extend `data_list` with a random subset to achieve the padding
            data_list = data_list + random.sample(data_list, padding_needed)
        # Shuffle the full list
        random.shuffle(data_list)
        return data_list        
    
    def _initialize_train_iterators(self):
        # Initialize padded train and validation data lists
        self.padded_train_list = self._get_padded_list(self.myRunIDs['train'], self.train_batch_size)

        # Create iterators
        self.train_data_iter = iter(self._get_data_batches(self.padded_train_list, self.train_batch_size))

    def _initialize_val_iterators(self):     
        # Initialize padded train and validation data lists
        self.padded_val_list = self._get_padded_list(self.myRunIDs['validate'], self.val_batch_size)

        # Create iterators
        self.val_data_iter = iter(self._get_data_batches(self.padded_val_list, self.val_batch_size))        

    def _get_data_batches(self, data_list, batch_size):
        # Generator that yields batches of specified size
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]

    def _extract_run_ids(self, data_iter_name, initialize_method):
        # Access the instance's data iterator by name
        data_iter = getattr(self, data_iter_name)
        try:
            # Attempt to get the next batch from the iterator
            runIDs = next(data_iter)
        except StopIteration:
            # Reinitialize the iterator if exhausted
            initialize_method()
            # Update the iterator reference after reinitialization
            data_iter = getattr(self, data_iter_name)
            runIDs = next(data_iter)
        # Update the instance attribute with the new iterator state
        setattr(self, data_iter_name, data_iter)
        return runIDs
    
    def create_train_dataloaders(
        self,
        crop_size: int = 96,
        num_samples: int = 16):
    
        train_batch_size = 1
        val_batch_size = 1

        # Non-random transforms to be cached
        non_random_transforms = Compose([
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS")
        ])

        # Random transforms to be applied during training
        random_transforms = Compose([
            # Geometric Transforms
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[crop_size, crop_size, crop_size],
                num_classes=self.Nclasses,
                num_samples=num_samples
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
        ])

        # Validation transforms
        val_transforms = Compose([
            RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[crop_size, crop_size, crop_size],
                    num_classes=self.Nclasses,
                    num_samples=num_samples, 
            ),
        ])

        # Augmentations to Explore in the Future: 
        # Intensity-based augmentations
        # RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),
        # RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.9, 1.1)),
        # RandScaleIntensityd(keys="image", factors=(0.9, 1.1), prob=0.5),
        # RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
        # RandGaussianSmoothd(keys="image", prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        # RandHistogramShiftd(keys="image", prob=0.5, num_control_points=(3, 5))

        # Geometric Transforms
        # RandAffined(
        #     keys=["image", "label"],
        #     rotate_range=(0.1, 0.1, 0.1),  # Rotation angles (radians) for x, y, z axes
        #     scale_range=(0.1, 0.1, 0.1),   # Scale range for isotropic/anisotropic scaling
        #     prob=0.5,                      # Probability of applying the transform
        #     padding_mode="border"          # Handle out-of-bounds values
        # )

        # We Only Need to Reload the Training Dataset if the Total Number of Runs is larger than 
        # the tomo batch size
        if self.train_loader is None: 

            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = io.load_training_data(self.root, trainRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id, 
                                                progress_update=False)

            # Create the cached dataset with non-random transforms
            train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_train_dataset = dataset.DynamicDataset(data=train_ds, transform=random_transforms)

            # DataLoader remains the same
            self.train_loader = DataLoader(
                self.dynamic_train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

        else:
            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = io.load_training_data(self.root, trainRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id, 
                                                progress_update=False)

            train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)
            self.dynamic_train_dataset.update_data(train_ds)

        # We Only Need to Reload the Validation Dataset if the Total Number of Runs is larger than 
        # the tomo batch size
        if self.reload_validation_dataset or self.val_loader is None: 

            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files   = io.load_training_data(self.root, validateRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id,
                                                progress_update=False)    

            # Create validation dataset
            val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_validation_dataset = dataset.DynamicDataset(data=val_ds, transform=val_transforms)

            # Create validation DataLoader
            self.val_loader  = DataLoader(
                self.dynamic_validation_dataset,
                batch_size=val_batch_size,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                shuffle=False,  # Ensure the data order remains consistent
            )
        else:
            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files   = io.load_training_data(self.root, validateRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id,
                                                progress_update=False)  

        return self.train_loader, self.val_loader
    
    def get_reload_frequency(self, num_epochs: int):
        """
        Automatically calculate the reload frequency for the dataset during training.

        Returns:
            int: Reload frequency (number of epochs between dataset reloads).
        """
        if not self.reload_training_dataset:
            # No need to reload if all tomograms fit in memory
            print("All training samples fit in memory. No reloading required.")
            self.reload_frequency = -1

        else:
            # Calculate the number of segments based on total training runs and batch size
            num_segments = (len(self.myRunIDs['train']) + self.tomo_batch_size - 1) // self.tomo_batch_size

            # Calculate reload frequency to distribute reloading evenly over epochs
            self.reload_frequency = max(num_epochs // num_segments, 1)

            # Warn if the number of epochs is insufficient for full dataset coverage
            if num_epochs < num_segments:
                print(
                    f"Warning: Chosen number of epochs ({num_epochs}) may not be sufficient "
                    f"to train over all training samples. Consider increasing the number of epochs "
                    f"to at least {num_segments}."
                )

    def get_dataloader_parameters(self):

        parameters = {
            'config': self.copick_config,
            'target_name': self.target_name,
            'target_session_id': self.target_session_id,
            'target_user_id': self.target_user_id,
            'voxel_size': self.voxel_size,
            'tomo_algorithm': self.tomo_algorithm,
            'tomo_batch_size': self.tomo_batch_size,
            'reload_frequency': self.reload_frequency,
            'testRunIDs': self.myRunIDs['test'],
            'valRunIDs': self.myRunIDs['validate'],    
            'trainRunIDs': self.myRunIDs['train'],
        }

        return parameters                

class PredictLoaderManager:

    def __init__(self, 
                 config: str, 
                 voxel_size: float = 10, 
                 tomo_algorithm: str = 'wbp', 
                 tomo_batch_size: int = 15, # Number of Tomograms to Load Per Sub-Epoch    
                 Nclasses: int = 3):
        
        # Read Copick Project
        self.copick_config = config
        self.root = io.load_copick_config(config)

        # Copick Query For Input Tomogram
        self.voxel_size = voxel_size
        self.tomo_algorithm = tomo_algorithm

        self.Nclasses = Nclasses
        self.tomo_batch_size = tomo_batch_size 


    def create_predict_dataloader(
        self, 
        voxel_spacing: float, 
        tomo_algorithm: str,       
        runIDs: str = None):

        # define pre transforms
        pre_transforms = Compose(
            [   EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                NormalizeIntensityd(keys=["image"]),
        ])

        # Split trainRunIDs, validateRunIDs, testRunIDs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs]
        test_files = io.load_predict_data(self.root, runIDs, voxel_spacing, tomo_algorithm)  

        test_ds = CacheDataset(data=test_files, transform=pre_transforms)
        test_loader = DataLoader(test_ds, 
                                batch_size=4, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=torch.cuda.is_available())
        return test_loader
    
    def get_dataloader_parameters(self):

        parameters = {
            'config': self.copick_config,
            'voxel_size': self.voxel_size,
            'tomo_algorithm': self.tomo_algorithm
        }

        return parameters    
    
    