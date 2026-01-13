from monai.data import DataLoader, SmartCacheDataset, CacheDataset
from octopi.datasets import helpers as utils
from monai.transforms import Compose
from octopi.datasets import augment
from octopi.datasets import io
import torch

class CopickDataModule:
    def __init__(self, 
                 config: str, 
                 tomo_alg: str,
                 name: str,
                 sessionid: str = None,
                 userid: str = None,
                 voxel_size: float = 10, 
                 tomo_batch_size: int = 15,
                 bgr: float = 0.0
                 ): 

        # Read Copick Projectdd
        self.config = config
        self.root = io.load_copick_config(config)

        # Member Variables
        self.target_name = name
        self.target_session_id = sessionid
        self.target_user_id = userid
        self.voxel_size = voxel_size
        self.tomo_alg = tomo_alg.split(",")
        self.tomo_batch_size = tomo_batch_size        
        self.bgr = bgr

        # Construct the Target URI
        self.target_uri = utils.build_target_uri(name, sessionid, userid, voxel_size)

        # Initialize the input dimensions   
        self.nx, self.ny, self.nz = None, None, None

        # Available Run IDs
        self.allRunIDs = self.get_available_runs()

    def get_available_runs(self):
        """
        Identify and return a list of run IDs that have segmentations available for the target.
        
        - Iterates through all runs in the project to check for segmentations that match 
        the specified target name, session ID, and user ID.
        - Only includes runs that have at least one matching segmentation.

        Returns:
            available_runIDs (list): List of run IDs with available segmentations.
        """

        # Get the requested tomogram algorithms
        requested_algs = set(self.tomo_alg) if self.tomo_alg else set()
 
        # Scan the Runs
        available, runs_with_seg, algs_present = utils.scan_runs(
            root=self.root,
            target_name=self.target_name,
            target_session_id=self.target_session_id,
            target_user_id=self.target_user_id,
            voxel_size=self.voxel_size,
            requested_algs=requested_algs,
        )

        # If There are Missing Tomogram Algorithms or Segmentations, Inform the User
        missing = requested_algs - algs_present
        if runs_with_seg == 0:
            utils.missing_segmentations(self.target_name, self.target_session_id, self.target_user_id)
        elif missing: 
            utils.missing_tomograms(missing)

        return available

    def get_data_splits(
        self, 
        trainRunIDs: list[str] = None, validateRunIDs: list[str] = None,
        train_ratio: float = 0.8, val_ratio: float = 0.2, test_ratio: float = 0.0,
        create_test_dataset: bool = False
        ):
        """
        Get the data splits.
        """

        # Get the Data Splits
        self.myRunIDs = utils.get_data_splits(
            self.allRunIDs, trainRunIDs, validateRunIDs, 
            train_ratio, val_ratio, test_ratio, create_test_dataset
        )

        # Get Class Info from the Training Dataset
        target_info = (self.target_name, self.target_session_id, self.target_user_id)
        self.Nclasses, self.class_names = utils.get_class_info(
            self.config, self.myRunIDs['train'].keys(), target_info, self.voxel_size )

        return self.myRunIDs
    
    def create(self, 
        crop_size: int = 96,
        num_samples: int = 64,
        train_transforms: Compose = None,
        val_transforms: Compose = None,
        val_batch_size: int = 1
        ):
        """
        Create the training and validation datasets and return the DataLoaders.

        Args:
            crop_size (int): The size of the crop to use for the training and validation sets.
            num_samples (int): The number of samples to use for the training and validation sets.

        Returns:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.        
        """

        # Define the Input Dimensions
        self.input_dim = crop_size, crop_size, crop_size

        # Create the list of training files
        train_files = [
            { 'run': run_name, 'root': self.config, 
                'vol_uri': f'{alg}@{self.voxel_size}', 
                'target_uri': self.target_uri }
            for run_name, algs in self.myRunIDs['train'].items()
            for alg in algs
        ]

        # Default Train Transforms for Particle Picking
        if train_transforms is None:
            train_transforms = Compose([
                augment.get_transforms(),
                augment.get_random_transforms(self.input_dim, num_samples, self.Nclasses, self.bgr)
            ])

        # Use SmartCacheDataset if the number of training files exceeds the tomo_batch_size
        if len(train_files) > self.tomo_batch_size:
            self.train_ds = SmartCacheDataset(
                data=train_files,                
                transform=train_transforms,
                cache_num=self.tomo_batch_size,  # e.g. 8–64 volumes
                replace_rate=0.3,                # e.g. 0.2–0.3
                num_init_workers=8,
                num_replace_workers=8,
                shuffle=False,
            )
        else:
            self.train_ds = CacheDataset(
                data=train_files,                
                transform=train_transforms,
                cache_rate=1.0,          # cache all items
            )

        # Create the DataLoader
        train_loader = DataLoader(
            self.train_ds, batch_size=1, 
            shuffle=True, num_workers=8, 
            pin_memory=torch.cuda.is_available()
        )         

        # Create the list of validation files
        val_files = [
            { 'run': run_name, 'root': self.config, 
                'vol_uri': f'{alg}@{self.voxel_size}', 
                'target_uri': self.target_uri }
            for run_name, algs in self.myRunIDs['validate'].items()
            for alg in algs
        ]

        # Default Val Transforms for Particle Picking
        if val_transforms is None:
            val_transforms = augment.get_transforms()

        # Create the CacheDataset
        val_ds = CacheDataset(
            data=val_files,                
            transform=val_transforms,
            cache_rate=1.0,          # cache all val items
            num_workers=8,           # threads for initial caching
        )   

        # Create the DataLoader
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size, 
            shuffle=False, num_workers=0, 
            pin_memory=torch.cuda.is_available()
        )

        # Print the data splits
        utils.print_splits(self.myRunIDs, train_files, val_files)        

        return train_loader, val_loader

    def get_dataloader_parameters(self):
        """
        Get the datamodule parameters.
        """
        return utils.get_parameters(self)

########################################################################################

class MultiCopickDataModule:
    def __init__(self,
                 configs: dict[str, str],
                 tomo_alg: str,
                 name: str,
                 sessionid: str = None,
                 userid: str = None,
                 voxel_size: float = 10,
                 tomo_batch_size: int = 15,
                 bgr: float = 0.0):
        """
        Initialize MutliCopickDataModule with multiple configs.

        Args:
            configs (list): List of config file paths.
            Other arguments are inherited from TrainLoaderManager.
        """
        # Read Copick Projects
        self.config = configs
        self.roots = {name: io.load_copick_config(path) for name, path in configs.items()}

        # Member Varialbles
        self.target_name = name
        self.target_session_id = sessionid
        self.target_user_id = userid
        self.voxel_size = voxel_size
        self.tomo_alg = tomo_alg.split(",")
        self.tomo_batch_size = tomo_batch_size
        self.bgr = bgr
        
        # Construct the Target URI
        self.target_uri = utils.build_target_uri(name, sessionid, userid, voxel_size)

        # Initialize the input dimensions   
        self.nx, self.ny, self.nz = None, None, None

        # Available Run IDs
        self.allRunIDs = self.get_available_runs()

    def get_available_runs(self):
        """
        Identify and return a list of run IDs that have segmentations available for the target.
        """
        requested_algs = {a.strip() for a in self.tomo_alg if a.strip()}
        all_available: dict[str, dict[str, list[str]]] = {}

        total_runs_with_seg = 0
        algs_present_global: set[str] = set()

        # Track per-session diagnostics
        session_runs_with_seg: dict[str, int] = {}
        session_algs_present: dict[str, set[str]] = {}

        for session_key, root in self.roots.items():
            available, runs_with_seg, algs_present = utils.scan_runs(
                root=root,
                target_name=self.target_name,
                target_session_id=self.target_session_id,
                target_user_id=self.target_user_id,
                voxel_size=self.voxel_size,
                requested_algs=requested_algs,
            )

            # `available` here is {run_id: [algs]} for that root
            if available:
                all_available[session_key] = available

            total_runs_with_seg += runs_with_seg
            algs_present_global |= algs_present

            session_runs_with_seg[session_key] = runs_with_seg
            session_algs_present[session_key] = algs_present

        # 1) No segmentations anywhere => hard error
        if total_runs_with_seg == 0:
            utils.missing_segmentations(self.target_name, self.target_session_id, self.target_user_id)

        # 2) Requested tomograms missing globally => warning
        if requested_algs:
            missing_global = requested_algs - algs_present_global
            if missing_global:
                utils.missing_tomograms(missing_global)

            # 3) Helpful per-session warnings
            for session_key, runs_with_seg in session_runs_with_seg.items():
                # Only warn if that session actually had segs (otherwise it's not a tomo-alg issue)
                if runs_with_seg > 0:
                    missing_here = requested_algs - session_algs_present[session_key]
                    if missing_here == requested_algs:
                        print(
                            f"\n[Warning] Config '{session_key}' has matching segmentations, "
                            f"but none of the requested tomo algs are present: {sorted(requested_algs)}\n"
                        )

        return all_available

    def get_data_splits(
        self, 
        trainRunIDs: list[str] = None, validateRunIDs: list[str] = None,
        train_ratio: float = 0.8, val_ratio: float = 0.2, test_ratio: float = 0.0,
        create_test_dataset: bool = False
        ):
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

        # Initialize the Run IDs
        per_session_splits: dict[str, dict[str, dict[str, list[str]]]] = {}

        for session_key, runIDs in self.allRunIDs.items():
            per_session_splits[session_key] = utils.get_data_splits(
                runIDs, 
                trainRunIDs=trainRunIDs, 
                validateRunIDs=validateRunIDs, 
                train_ratio=train_ratio, val_ratio=val_ratio, 
                test_ratio=test_ratio, 
                create_test_dataset=create_test_dataset
            )

        # Merge into the nested-by-session format
        self.myRunIDs = {
            "train": {}, "validate": {}, "test": {},
        }

        for split_name in ("train", "validate", "test"):
            self.myRunIDs[split_name] = {
                session_key: per_session_splits[session_key][split_name]
                for session_key in per_session_splits.keys()
            }         

        # ---- Class info ----
        # Pick one session/config as the "label schema source".
        # (Assumes exp/sim share the same target YAML label map.)
        target_info = (self.target_name, self.target_session_id, self.target_user_id)
        first_session = next(iter(self.config.keys()))
        config_path = self.config[first_session]

        # Get Class Info from the Training Dataset
        self.Nclasses, self.class_names = utils.get_class_info(
            config_path, self.myRunIDs['train'][first_session].keys(), target_info, self.voxel_size )

        return self.myRunIDs

    def create(self, 
        crop_size: int = 96,
        num_samples: int = 64,
        train_transforms: Compose = None,
        val_transforms: Compose = None,
        val_batch_size: int = 1,
        ):
        """
        Create the training and validation datasets and return the DataLoaders.
        """
        # Define the Input Dimensions
        self.input_dim = crop_size, crop_size, crop_size

        # Create the list of training files
        train_files = [
            { 'run': run_id, 
               "root": self.config[session_key], 
              'vol_uri': f'{alg}@{self.voxel_size}', 
              'target_uri': self.target_uri }
            for session_key, runmap in self.myRunIDs["train"].items()
            for run_id, algs in runmap.items()
            for alg in algs
        ]

        # Default Train Transforms for Particle Picking
        if train_transforms is None:
            train_transforms = Compose([
                augment.get_transforms(),
                augment.get_random_transforms(self.input_dim, num_samples, self.Nclasses, self.bgr)
            ])

        # Create the SmartCacheDataset
        self.train_ds = SmartCacheDataset(
            data=train_files,                
            transform=train_transforms,
            cache_num=self.tomo_batch_size,  # e.g. 8–64 volumes
            replace_rate=0.3,                # e.g. 0.2–0.3
            num_init_workers=8,
            num_replace_workers=8,
            shuffle=False,
        )

        # Create the DataLoader
        train_loader = DataLoader(
            self.train_ds, batch_size=1, 
            shuffle=True, num_workers=8, 
            pin_memory=torch.cuda.is_available()
        )         

        # Create the list of validation files
        val_files = [
            { 'run': run_id, 
              'root': self.config[session_key], 
              'vol_uri': f'{alg}@{self.voxel_size}', 
              'target_uri': self.target_uri }
            for session_key, runmap in self.myRunIDs["validate"].items()
            for run_id, algs in runmap.items()
            for alg in algs                
        ]

        # Default Val Transforms for Particle Picking
        if val_transforms is None:
            val_transforms = augment.get_transforms()

        # Create the CacheDataset
        val_ds = CacheDataset(
            data=val_files,                
            transform=val_transforms,
            cache_rate=1.0,          # cache all val items
            num_workers=8,           # threads for initial caching
        )   

        # Create the DataLoader
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size, 
            shuffle=False, num_workers=0, 
            pin_memory=torch.cuda.is_available()
        )

        # Print the data splits
        utils.print_splits(self.myRunIDs, train_files, val_files)

        return train_loader, val_loader

    def get_dataloader_parameters(self):
        """
        Get the datamodule parameters.
        """
        return utils.get_parameters(self) 