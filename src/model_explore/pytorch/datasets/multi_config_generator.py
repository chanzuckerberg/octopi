from model_explore.pytorch.datasets.generators import TrainLoaderManager
from monai.data import DataLoader, CacheDataset, Dataset
from model_explore.pytorch.datasets import dataset
from model_explore.pytorch import io
import torch, random

class MultiConfigTrainLoaderManager(TrainLoaderManager):

    def __init__(self, 
                 configs: list,  # List of config file paths
                 target_name: str,
                 target_session_id: str = None,
                 target_user_id: str = None,
                 voxel_size: float = 10, 
                 tomo_algorithm: str = 'wbp', 
                 tomo_batch_size: int = 15, 
                 Nclasses: int = 3):
        """
        Initialize MultiConfigTrainLoaderManager with multiple configs.

        Args:
            configs (list): List of config file paths.
            Other arguments are inherited from TrainLoaderManager.
        """
        # Call the parent constructor
        super().__init__(
            config=configs[0],  # Use the first config for initialization
            target_name=target_name,
            target_session_id=target_session_id,
            target_user_id=target_user_id,
            voxel_size=voxel_size,
            tomo_algorithm=tomo_algorithm,
            tomo_batch_size=tomo_batch_size,
            Nclasses=Nclasses
        )

        # Load multiple roots for the configs
        self.roots = {}
        for idx, config in enumerate(configs):
            self.roots[f'config_{idx}'] = io.load_copick_config(config)
        
        # Update the original root to include all loaded roots
        self.root = self.roots

    def get_data_splits(self,
                        trainRunIDs: dict = None,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        create_test_dataset: bool = True):
        """
        Split datasets from multiple roots into train, validate, and test splits.

        Args:
            trainRunIDs (dict): Dictionary of train run IDs for each root.
            validateRunIDs (dict): Dictionary of validation run IDs for each root.
            Other arguments are inherited from TrainLoaderManager.
        """
        all_trainRunIDs = []
        all_validateRunIDs = []
        all_testRunIDs = []

        for root_key, root in self.roots.items():
            runIDs = [run.name for run in root.runs]

            # Split data for the current root
            if trainRunIDs and root_key in trainRunIDs:
                train_run, validate_run, test_run = io.split_datasets(
                    trainRunIDs[root_key], train_ratio, val_ratio, test_ratio, create_test_dataset
                )
            else:
                train_run, validate_run, test_run = io.split_datasets(
                    runIDs, train_ratio, val_ratio, test_ratio, create_test_dataset
                )

            # Add to the combined lists
            all_trainRunIDs.extend(train_run)
            all_validateRunIDs.extend(validate_run)
            all_testRunIDs.extend(test_run)

        # Combine and shuffle the datasets
        random.shuffle(all_trainRunIDs)
        random.shuffle(all_validateRunIDs)
        random.shuffle(all_testRunIDs)

        # Assign combined run IDs
        self.myRunIDs = {
            'train': all_trainRunIDs,
            'validate': all_validateRunIDs,
            'test': all_testRunIDs
        }

        print(f"Total number of training samples: {len(all_trainRunIDs)}")
        print(f"Total number of validation samples: {len(all_validateRunIDs)}")
        print(f'Total number of test samples: {len(all_testRunIDs)}')

        self.train_batch_size = self.tomo_batch_size
        self.val_batch_size = min(len(all_validateRunIDs), self.tomo_batch_size)

        self._initialize_val_iterators()
        self._initialize_train_iterators()

        return self.myRunIDs

    def create_train_dataloaders(self, crop_size: int = 96, num_samples: int = 16):
        """
        Override the data loader creation to handle multiple roots.
        """
        if self.train_loader is None:
            # Fetch the next batch of run IDs
            trainRunIDs = self._extract_run_ids('train_data_iter', self._initialize_train_iterators)
            
            # Collect data from all roots
            train_files = []
            for root_key, root in self.roots.items():
                train_files.extend(io.load_training_data(
                    root, trainRunIDs, self.voxel_size, self.tomo_algorithm,
                    self.target_name, self.target_session_id, self.target_user_id, progress_update=False
                ))

            # Create the cached dataset with non-random transforms
            train_ds = CacheDataset(data=train_files, transform=self._get_transforms(), cache_rate=1.0)

            # Wrap the cached dataset to apply random transforms during iteration
            self.dynamic_train_dataset = dataset.DynamicDataset(data=train_ds, transform=self._get_random_transforms(crop_size, num_samples))

            # Create the DataLoader
            self.train_loader = DataLoader(
                self.dynamic_train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
        return self.train_loader