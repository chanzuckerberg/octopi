from model_explore.datasets.generators import TrainLoaderManager
from monai.data import DataLoader, CacheDataset, Dataset
from model_explore.datasets import dataset, augment
from model_explore import io
import multiprocess as mp
from typing import List
from tqdm import tqdm
import torch

class MultiConfigTrainLoaderManager(TrainLoaderManager):

    def __init__(self, 
                 configs: dict,  # Dictionary of session names and config paths
                 target_name: str,
                 target_session_id: str = None,
                 target_user_id: str = None,
                 voxel_size: float = 10, 
                 tomo_algorithm: List[str] = ['wbp'], 
                 tomo_batch_size: int = 15, 
                 Nclasses: int = 3):
        """
        Initialize MultiConfigTrainLoaderManager with multiple configs.

        Args:
            configs (list): List of config file paths.
            Other arguments are inherited from TrainLoaderManager.
        """

        # Initialize shared attributes manually (skip super().__init__ to avoid invalid config handling)
        self.config = configs
        self.roots = {name: io.load_copick_config(path) for name, path in configs.items()}

        # Target and algorithm parameters
        self.target_name = target_name
        self.target_session_id = target_session_id
        self.target_user_id = target_user_id
        self.voxel_size = voxel_size
        self.tomo_algorithm = tomo_algorithm

        # Data management parameters
        self.Nclasses = Nclasses
        self.tomo_batch_size = tomo_batch_size
        self.reload_training_dataset = True
        self.reload_validation_dataset = True
        self.val_loader = None
        self.train_loader = None

        # Initialize Run IDs placeholder
        self.myRunIDs = {}

    def get_available_runIDs(self):
        """
        Identify and return a combined list of run IDs with available segmentations 
        across all configured CoPick projects.

        Returns:
            List of tuples: [(session_name, run_name), ...]
        """
        available_runIDs = []  
        for name, root in self.roots.items():
            runIDs = [run.name for run in root.runs]             
            for run in runIDs:
                run = root.get_run(run)
                seg = run.get_segmentations(
                    name=self.target_name, 
                    session_id=self.target_session_id, 
                    user_id=self.target_user_id,
                    voxel_size=self.voxel_size
                )
                if len(seg) > 0:
                    available_runIDs.append((name, run.name))  # Include session name for disambiguation
        
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
        Override to handle run IDs as (session_name, run_name) tuples.
        """
        # Use the get_available_runIDs method to handle multiple projects
        runIDs = self.get_available_runIDs()
        return super().get_data_splits(trainRunIDs = runIDs, 
                                       train_ratio = train_ratio, 
                                       val_ratio = val_ratio, 
                                       test_ratio = test_ratio)

    def _initialize_train_iterators(self):
        """
        Initialize the training data iterators with multi-config support.
        """
        self.padded_train_list = self._get_padded_list(self.myRunIDs['train'], self.train_batch_size)
        self.train_data_iter = iter(self._get_data_batches(self.padded_train_list, self.train_batch_size))

    def _initialize_val_iterators(self):
        """
        Initialize the validation data iterators with multi-config support.
        """
        self.padded_val_list = self._get_padded_list(self.myRunIDs['validate'], self.val_batch_size)
        self.val_data_iter = iter(self._get_data_batches(self.padded_val_list, self.val_batch_size))

    def _load_data(self, runIDs):
        """
        Load data from multiple CoPick projects for given run IDs.

        Args:
            runIDs (list): List of (session_name, run_name) tuples.

        Returns:
            List: Combined dataset for the specified run IDs.
        """
        data = []
        for session_name, run_name in tqdm(runIDs):
            root = self.roots[session_name]
            data.extend(io.load_training_data(
                root, [run_name], self.voxel_size, self.tomo_algorithm, 
                self.target_name, self.target_session_id, self.target_user_id, 
                progress_update=False ))
        self._check_max_label_value(data)
        return data

    def create_train_dataloaders(self, *args, **kwargs):
        """
        Override data loading to fetch from multiple projects.
        """
        my_crop_size = kwargs.get("crop_size", 96)
        my_num_samples = kwargs.get("num_samples", 16)

        train_batch_size = 1
        val_batch_size = 1  

        # If reloads are disabled and loaders already exist, reuse them
        if self.reload_frequency < 0 and (self.train_loader is not None) and (self.val_loader is not None):
            return self.train_loader, self.val_loader        

        # Estimate Max Number of Threads with mp.cpu_count
        n_procs = min(mp.cpu_count(), 4)               

        if self.train_loader is None: 
            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = self._load_data(trainRunIDs)

            # Create the cached dataset with non-random transforms
            train_ds = CacheDataset(data=train_files, transform=augment.get_transforms(), cache_rate=1.0)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_train_dataset = dataset.DynamicDataset(
                data=train_ds, 
                transform=augment.get_random_transforms(my_crop_size, my_num_samples, self.Nclasses)
            )           

            self.train_loader = DataLoader(
                self.dynamic_train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=n_procs,
                pin_memory=torch.cuda.is_available()
            )

        else:
            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            train_files = self._load_data(trainRunIDs)
            train_ds = CacheDataset(data=train_files, transform=augment.get_transforms(), cache_rate=1.0)
            self.dynamic_train_dataset.update_data(train_ds)

        # We Only Need to Reload the Validation Dataset if the Total Number of Runs is larger than 
        # the tomo batch size
        if self.val_loader is None: 

            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files  = self._load_data(validateRunIDs)    

            # Create validation dataset
            val_ds = CacheDataset(data=val_files, transform=augment.get_transforms(), cache_rate=1.0)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_validation_dataset = dataset.DynamicDataset(
                data=val_ds, 
                transform=augment.get_validation_transforms(my_crop_size, my_num_samples, self.Nclasses) 
            )

            # Create validation DataLoader
            self.val_loader  = DataLoader(
                self.dynamic_validation_dataset,
                batch_size=val_batch_size,
                num_workers=n_procs,
                pin_memory=torch.cuda.is_available(),
                shuffle=False,  # Ensure the data order remains consistent
            )
        else:
            validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)             
            val_files   = self._load_data(validateRunIDs)

            val_ds = CacheDataset(data=val_files, transform=augment.get_transforms(), cache_rate=1.0)
            self.dynamic_validation_dataset.update_data(val_ds)
            
        return self.train_loader, self.val_loader


    def tmp_return_datasets(self):
        trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
        train_files = self._load_data(trainRunIDs)   

        validateRunIDs = self._extract_run_ids('val_data_iter', self._initialize_val_iterators)
        val_files = self._load_data(validateRunIDs)

        return train_files, val_files        