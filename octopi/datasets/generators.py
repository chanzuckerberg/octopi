from octopi.datasets import dataset, augment, cached_datset
from monai.data import DataLoader, SmartCacheDataset, CacheDataset, Dataset
from typing import List, Optional
from octopi.datasets import io
import torch, os, random, gc
import multiprocess as mp

class TrainLoaderManager:

    def __init__(self, 
                 config: str, 
                 target_name: str,
                 target_session_id: str = None,
                 target_user_id: str = None,
                 voxel_size: float = 10, 
                 tomo_algorithm: List[str] = ['wbp'], 
                 tomo_batch_size: int = 15, # Number of Tomograms to Load Per Sub-Epoch    
                 Nclasses: int = 3): # Number of Objects + Background

        # Read Copick Projectdd
        self.config = config
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

        # Initialize the input dimensions   
        self.nx = None
        self.ny = None
        self.nz = None

    def get_available_runIDs(self):
        """
        Identify and return a list of run IDs that have segmentations available for the target.
        
        - Iterates through all runs in the project to check for segmentations that match 
        the specified target name, session ID, and user ID.
        - Only includes runs that have at least one matching segmentation.

        Returns:
            available_runIDs (list): List of run IDs with available segmentations.
        """        
        available_runIDs = []
        runIDs = [run.name for run in self.root.runs]
        for run in runIDs:
            run = self.root.get_run(run)
            seg = run.get_segmentations(name=self.target_name, 
                                        session_id=self.target_session_id, 
                                        user_id=self.target_user_id,
                                        voxel_size=float(self.voxel_size))
            if len(seg) > 0:
                available_runIDs.append(run.name)

        # If No Segmentations are Found, Inform the User
        if len(available_runIDs) == 0:
            print(
                f"[Error] No segmentations found for the target query:\n"
                f"TargetName: {self.target_name}, UserID: {self.target_user_id}, "
                f"SessionID: {self.target_session_id}\n"
                f"Please check the target name, user ID, and session ID.\n"
            )
            exit()

        return available_runIDs

    def get_data_splits(self,
                        trainRunIDs: str = None,
                        validateRunIDs: str = None,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        create_test_dataset: bool = True):
        """
        Split the available data into training, validation, and testing sets based on input parameters.

        Args:
            trainRunIDs (str): Predefined list of run IDs for training. If provided, it overrides splitting logic.
            validateRunIDs (str): Predefined list of run IDs for validation. If provided with trainRunIDs, no splitting occurs.
            train_ratio (float): Proportion of available data to allocate to the training set.
            val_ratio (float): Proportion of available data to allocate to the validation set.
            test_ratio (float): Proportion of available data to allocate to the test set.
            create_test_dataset (bool): Whether to create a test dataset or leave it empty.

        Returns:
            myRunIDs (dict): Dictionary containing run IDs for training, validation, and testing.
        """          

        # Option 1: Only TrainRunIDs are Provided, Split into Train, Validate and Test (Optional)
        if trainRunIDs is not None and validateRunIDs is None:
            trainRunIDs, validateRunIDs, testRunIDs = io.split_multiclass_dataset(
                trainRunIDs, train_ratio, val_ratio, test_ratio, 
                return_test_dataset = create_test_dataset
            )
        # Option 2: TrainRunIDs and ValidateRunIDs are Provided, No Need to Split
        elif trainRunIDs is not None and validateRunIDs is not None:
            testRunIDs = None
        # Option 3: Use the Entire Copick Project, Split into Train, Validate and Test
        else:
            runIDs = self.get_available_runIDs()          
            trainRunIDs, validateRunIDs, testRunIDs = io.split_multiclass_dataset(
                runIDs, train_ratio, val_ratio, test_ratio, 
                return_test_dataset = create_test_dataset
            )

        # Swap if Test Runs is Larger than Validation Runs
        if create_test_dataset and len(testRunIDs) > len(validateRunIDs):
            testRunIDs, validateRunIDs = validateRunIDs, testRunIDs

        # Determine if datasets fit entirely in memory based on the batch size
        # If the validation set is smaller than the batch size, avoid reloading
        if len(validateRunIDs) < self.tomo_batch_size:
            self.reload_validation_dataset  = False

        # If the training set is smaller than the batch size, avoid reloading
        if len(trainRunIDs) < self.tomo_batch_size:
            self.reload_training_dataset = False

        # Store the split run IDs into a dictionary for easy access
        self.myRunIDs = {
            'train': trainRunIDs,
            'validate': validateRunIDs,
            'test': testRunIDs
        }

        print(f"Number of training samples: {len(trainRunIDs)}")
        print(f"Number of validation samples: {len(validateRunIDs)}")
        if testRunIDs is not None:
            print(f'Number of test samples: {len(testRunIDs)}')    

        # Define separate batch sizes
        self.train_batch_size = min( len(self.myRunIDs['train']), self.tomo_batch_size)
        self.val_batch_size   = min( len(self.myRunIDs['validate']), self.tomo_batch_size)

        # Initialize data iterators for training and validation
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
        num_samples: int = 64):

        train_batch_size = 1
        val_batch_size = 1

        # If reloads are disabled and loaders already exist, reuse them
        if self.reload_frequency < 0 and (self.train_loader is not None) and (self.val_loader is not None):
            return self.train_loader, self.val_loader         

        # We Only Need to Reload the Training Dataset if the Total Number of Runs is larger than 
        # the tomo batch size
        if self.train_loader is None: 

            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = io.load_training_data(self.root, trainRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id, 
                                                progress_update=False)
            self._check_max_label_value(train_files)

            # Create the cached dataset with non-random transforms
            train_ds = CacheDataset(data=train_files, transform=augment.get_transforms(), cache_rate=1.0)

            # Delete the training files to free memory
            train_files = None
            gc.collect()

            # I need to read (nx,ny,nz) and scale the crop size to make sure it isnt larger than nx.
            if self.nx is None: (self.nx,self.ny,self.nz) = train_ds[0]['image'].shape[1:]
            self.input_dim = io.get_input_dimensions(train_ds, crop_size)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_train_dataset = dataset.DynamicDataset(
                data=train_ds, 
                transform=augment.get_random_transforms(self.input_dim, num_samples, self.Nclasses)
            )

            # Define the number of processes for the DataLoader
            n_procs = min(mp.cpu_count(), 4)

            # DataLoader remains the same
            self.train_loader = DataLoader(
                self.dynamic_train_dataset,
                batch_size=train_batch_size,
                shuffle=False,
                num_workers=n_procs,
                pin_memory=torch.cuda.is_available(),
            )

        else:
            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = io.load_training_data(self.root, trainRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id, 
                                                progress_update=False)
            self._check_max_label_value(train_files)

            train_ds = CacheDataset(data=train_files, transform=augment.get_transforms(), cache_rate=1.0)
            self.dynamic_train_dataset.update_data(train_ds)

        # We Only Need to Reload the Validation Dataset if the Total Number of Runs is larger than 
        # the tomo batch size
        if self.val_loader is None: 

            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files   = io.load_training_data(self.root, validateRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id,
                                                progress_update=False) 
            self._check_max_label_value(val_files)

            # Create validation dataset
            val_ds = CacheDataset(data=val_files, transform=augment.get_transforms(), cache_rate=1.0)

            # Delete the validation files to free memory
            val_files = None
            gc.collect()

            # # I need to read (nx,ny,nz) and scale the crop size to make sure it isnt larger than nx.
            # if self.nx is None:
            #     (self.nx,self.ny,self.nz) = val_ds[0]['image'].shape[1:]

            # if crop_size > self.nx: self.input_dim = (self.nx, crop_size, crop_size)
            # else:                   self.input_dim = (crop_size, crop_size, crop_size)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_validation_dataset = dataset.DynamicDataset( data=val_ds )

            dataset_size = len(self.dynamic_validation_dataset)
            n_procs = min(mp.cpu_count(), 8)

            # Create validation DataLoader
            self.val_loader  = DataLoader(
                self.dynamic_validation_dataset,
                batch_size=val_batch_size,
                num_workers=n_procs,
                pin_memory=torch.cuda.is_available(),
                shuffle=False,  # Ensure the data order remains consistent,            
            )
        else:
            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files   = io.load_training_data(self.root, validateRunIDs, self.voxel_size, self.tomo_algorithm, 
                                                self.target_name, self.target_session_id, self.target_user_id,
                                                progress_update=False)  
            self._check_max_label_value(val_files)

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

            print(f"\nReloading {self.tomo_batch_size} tomograms every {self.reload_frequency} epochs\n")

            # Warn if the number of epochs is insufficient for full dataset coverage
            if num_epochs < num_segments:
                print(
                    f"Warning: Chosen number of epochs ({num_epochs}) may not be sufficient "
                    f"to train over all training samples. Consider increasing the number of epochs "
                    f"to at least {num_segments}\n."
                )

    def _check_max_label_value(self, train_files):
        max_label_value = max(file['label'].max() for file in train_files)
        if max_label_value > self.Nclasses:
            print(f"Warning: Maximum class label value {max_label_value} exceeds the number of classes {self.Nclasses}.")
            print("This may cause issues with the model's output layer.")
            print("Consider adjusting the number of classes or the label values in your data.\n")

    def get_dataloader_parameters(self):

        parameters = {
            'config': self.config,
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

        # Initialize the input dimensions   
        self.nx = None
        self.ny = None
        self.nz = None


    def create_predict_dataloader(
        self, 
        voxel_spacing: float, 
        tomo_algorithm: str,       
        runIDs: str = None):

        # Split trainRunIDs, validateRunIDs, testRunIDs
        if runIDs is None:
            runIDs = [run.name for run in self.root.runs]
            
        # Load the test data
        test_files = io.load_predict_data(self.root, runIDs, voxel_spacing, tomo_algorithm)  

        # Create the cached dataset with non-random transforms
        test_ds = CacheDataset(data=test_files, transform=augment.get_predict_transforms())

        # Read (nx,ny,nz) for input tomograms.
        if self.nx is None:
            (self.nx,self.ny,self.nz) = test_ds[0]['image'].shape[1:]

        # Create the DataLoader
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
    
    