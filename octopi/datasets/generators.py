from monai.data import (
    DataLoader, SmartCacheDataset, CacheDataset,
)
from octopi.datasets import helpers as utils
from monai.transforms import Compose
from octopi.datasets import augment
from octopi.datasets import io
from typing import List
import torch

class CopickDataModule:
    def __init__(self, 
                 config: str, 
                 tomo_uris: List[str],
                 name: str,
                 sessionid: str = None,
                 userid: str = None,
                 tomo_batch_size: int = 15,
                 bgr: float = 0.0,
                 verbose: bool = True
                 ): 

        # Read Copick Projectdd
        self.config = config
        self.root = io.load_copick_config(config)

        # Member Variables
        self.target_name = name
        self.target_session_id = sessionid
        self.target_user_id = userid
        self.tomo_batch_size = tomo_batch_size
        self.bgr = bgr
        self.verbose = verbose

        # Parse tomogram URIs into (alg, voxel_size) resolution pairs. Each pair
        # is an independent training source (multi-source training — mixing voxel
        # sizes and/or reconstruction algorithms); the target segmentation voxel
        # size is derived from each tomogram URI.
        self.tomo_uris = tomo_uris
        self.resolutions = utils.parse_resolution_uris(tomo_uris)

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
            available (dict): {run_id: [(alg, voxel_size), ...]} resolutions present per run.
        """

        # Scan the Runs for each requested (alg, voxel_size) resolution
        available, runs_with_seg, missing = utils.scan_runs(
            root=self.root,
            resolutions=self.resolutions,
            target_name=self.target_name,
            target_session_id=self.target_session_id,
            target_user_id=self.target_user_id,
        )

        # If There are Missing Resolutions or Segmentations, Inform the User
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

        # Get Class Info from the Training Dataset (classes are identical across
        # resolutions, so read from the first requested voxel size).
        target_info = (self.target_name, self.target_session_id, self.target_user_id)
        self.Nclasses, self.class_names = utils.get_class_info(
            self.config, self.myRunIDs['train'].keys(), target_info, self.resolutions[0][1] )

        return self.myRunIDs
    
    def create(self, 
        crop_size: int = 96,
        num_samples: int = 64,
        train_transforms: Compose = None,
        val_transforms: Compose = None,
        val_batch_size: int = 64
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

        # Create the list of training files. One entry per (run, resolution):
        # each (alg, voxel_size) pair is loaded at its native spacing with its
        # own matching target, so the model trains across all resolutions.
        train_files = [
            { 'run': run_name, 'root': self.config,
                'vol_uri': f'{alg}@{vs}',
                'target_uri': utils.build_target_uri(self.target_name, self.target_session_id, self.target_user_id, vs) }
            for run_name, pairs in self.myRunIDs['train'].items()
            for (alg, vs) in pairs
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
                cache_num=self.tomo_batch_size,
                replace_rate=0.15,
                num_init_workers=8,
                num_replace_workers=8,
                # shuffle=True so the cached window is a random mix of datasets rather than a
                # contiguous dataset-grouped sweep (train_files is grouped by run/dataset). With
                # shuffle=False the cache window slides through one acquisition at a time, the model
                # over-fits each in turn and partially forgets the rest -> a periodic sawtooth in
                # loss/val metrics (period ~= n_train_files / (replace_rate*cache_num)). The
                # DataLoader's shuffle only reorders WITHIN the small cached window, not which
                # tomograms are cached, so it does not substitute for this.
                shuffle=True,
            )
        else:
            self.train_ds = CacheDataset(
                data=train_files,                
                transform=train_transforms,
                cache_rate=1.0,          # cache all items
            )

        # Create the DataLoader
        #
        # persistent_workers=False is deliberate: with CacheDataset the
        # tomogram cache lives in the main process and is shared to workers
        # via copy-on-write. Over many epochs, workers drift toward their
        # own full copy of the cache (~cache_size per worker), which can
        # OOM on large worker counts. Re-forking each epoch resets COW.
        train_nw = utils.auto_num_workers()
        train_loader = DataLoader(
            self.train_ds, batch_size=1,
            shuffle=True, num_workers=train_nw,
            persistent_workers=False,
            prefetch_factor=4 if train_nw > 0 else None,
            pin_memory=torch.cuda.is_available()
        )

        # Create the list of validation files (one entry per run × resolution).
        val_files = [
            { 'run': run_name, 'root': self.config,
                'vol_uri': f'{alg}@{vs}',
                'target_uri': utils.build_target_uri(self.target_name, self.target_session_id, self.target_user_id, vs) }
            for run_name, pairs in self.myRunIDs['validate'].items()
            for (alg, vs) in pairs
        ]

        # Cache the FULL validation volumes (one item per run). The trainer
        # runs sliding_window_inference over each volume so the validation
        # metric matches inference-time behavior (overlap + Gaussian blending)
        # instead of scoring independent, zero-padded patches — the latter
        # systematically underestimates F1 by splitting particles at patch
        # boundaries. GPU memory stays bounded by the trainer's sw_batch_size
        # (only a few ROI windows resident at once), not by the volume size.
        val_ds = CacheDataset(data=val_files, transform=augment.get_transforms(), cache_rate=1.0, num_workers=4)

        # batch_size=1: validation is per full volume (volumes differ in shape
        # and cannot be stacked). Throughput is governed by the trainer's
        # sw_batch_size, not this loader. Few workers / no persistent_workers
        # so the cache RAM is reclaimed between infrequent validations.
        val_nw = max(1, train_nw // 4)
        val_loader = DataLoader(
            val_ds, batch_size=1,
            shuffle=False, num_workers=val_nw,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available()
        )

        # Print the data splits
        if self.verbose:
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
                 tomo_uris,
                 name: str,
                 sessionid: str = None,
                 userid: str = None,
                 tomo_batch_size: int = 15,
                 bgr: float = 0.0,
                 verbose: bool = True
                 ):
        """
        Initialize MutliCopickDataModule with multiple configs.

        Args:
            configs (list): List of config file paths.
            tomo_uris: Tomogram URI(s) (``alg@voxel_size``) for multi-source training — may mix
                voxel sizes and/or reconstruction algorithms.
            Other arguments are inherited from TrainLoaderManager.
        """
        # Read Copick Projects
        self.config = configs
        self.roots = {name: io.load_copick_config(path) for name, path in configs.items()}

        # Member Varialbles
        self.target_name = name
        self.target_session_id = sessionid
        self.target_user_id = userid
        self.tomo_batch_size = tomo_batch_size
        self.bgr = bgr
        self.verbose = verbose

        # Parse tomogram URIs into (alg, voxel_size) resolution pairs.
        self.tomo_uris = tomo_uris
        self.resolutions = utils.parse_resolution_uris(tomo_uris)

        # Initialize the input dimensions
        self.nx, self.ny, self.nz = None, None, None

        # Available Run IDs
        self.allRunIDs = self.get_available_runs()

    def get_available_runs(self):
        """
        Identify and return a list of run IDs that have segmentations available for the target.
        """
        requested = {(a, float(v)) for a, v in self.resolutions}
        all_available: dict[str, dict[str, list[tuple[str, float]]]] = {}

        total_runs_with_seg = 0
        resolutions_present_global: set[tuple[str, float]] = set()

        # Track per-session diagnostics
        session_runs_with_seg: dict[str, int] = {}
        session_resolutions_present: dict[str, set[tuple[str, float]]] = {}

        for session_key, root in self.roots.items():
            available, runs_with_seg, missing = utils.scan_runs(
                root=root,
                resolutions=self.resolutions,
                target_name=self.target_name,
                target_session_id=self.target_session_id,
                target_user_id=self.target_user_id,
            )

            # `available` here is {run_id: [(alg, voxel_size), ...]} for that root
            if available:
                all_available[session_key] = available

            total_runs_with_seg += runs_with_seg
            present_here = requested - missing
            resolutions_present_global |= present_here

            session_runs_with_seg[session_key] = runs_with_seg
            session_resolutions_present[session_key] = present_here

        # 1) No segmentations anywhere => hard error
        if total_runs_with_seg == 0:
            utils.missing_segmentations(self.target_name, self.target_session_id, self.target_user_id)

        # 2) Requested resolutions missing globally => warning
        missing_global = requested - resolutions_present_global
        if missing_global:
            utils.missing_tomograms(missing_global)

        # 3) Helpful per-session warnings
        for session_key, runs_with_seg in session_runs_with_seg.items():
            # Only warn if that session actually had segs (otherwise it's not a resolution issue)
            if runs_with_seg > 0 and not session_resolutions_present[session_key]:
                pretty = ", ".join(f"{a}@{v}" for a, v in sorted(requested))
                print(
                    f"\n[Warning] Config '{session_key}' has matching segmentations, "
                    f"but none of the requested resolutions are present: {pretty}\n"
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

        # Get Class Info from the Training Dataset (classes identical across
        # resolutions, so read from the first requested voxel size).
        self.Nclasses, self.class_names = utils.get_class_info(
            config_path, self.myRunIDs['train'][first_session].keys(), target_info, self.resolutions[0][1] )

        return self.myRunIDs

    def create(self, 
        crop_size: int = 96,
        num_samples: int = 64,
        train_transforms: Compose = None,
        val_transforms: Compose = None,
        val_batch_size: int = 64,
        ):
        """
        Create the training and validation datasets and return the DataLoaders.
        """
        # Define the Input Dimensions
        self.input_dim = crop_size, crop_size, crop_size

        # Create the list of training files (one entry per session × run × resolution).
        train_files = [
            { 'run': run_id,
               "root": self.config[session_key],
              'vol_uri': f'{alg}@{vs}',
              'target_uri': utils.build_target_uri(self.target_name, self.target_session_id, self.target_user_id, vs) }
            for session_key, runmap in self.myRunIDs["train"].items()
            for run_id, pairs in runmap.items()
            for (alg, vs) in pairs
        ]

        # Default Train Transforms for Particle Picking
        if train_transforms is None:
            train_transforms = Compose([
                augment.get_transforms(),
                augment.get_random_transforms(self.input_dim, num_samples, self.Nclasses, self.bgr)
            ])

        # Create the SmartCacheDataset.
        # shuffle=True so the cached window is a random mix of datasets, not a contiguous
        # dataset-grouped sweep (see the note in CopickDataModule.create).
        self.train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_num=self.tomo_batch_size,
            replace_rate=0.15,
            num_init_workers=8,
            num_replace_workers=8,
            shuffle=True,
        )

        # Create the DataLoader
        #
        # persistent_workers=False is deliberate: with CacheDataset the
        # tomogram cache lives in the main process and is shared to workers
        # via copy-on-write. Over many epochs, workers drift toward their
        # own full copy of the cache (~cache_size per worker), which can
        # OOM on large worker counts. Re-forking each epoch resets COW.
        train_nw = utils.auto_num_workers()
        train_loader = DataLoader(
            self.train_ds, batch_size=1,
            shuffle=True, num_workers=train_nw,
            persistent_workers=False,
            prefetch_factor=4 if train_nw > 0 else None,
            pin_memory=torch.cuda.is_available()
        )

        # Create the list of validation files (one entry per session × run × resolution).
        val_files = [
            { 'run': run_id,
              'root': self.config[session_key],
              'vol_uri': f'{alg}@{vs}',
              'target_uri': utils.build_target_uri(self.target_name, self.target_session_id, self.target_user_id, vs) }
            for session_key, runmap in self.myRunIDs["validate"].items()
            for run_id, pairs in runmap.items()
            for (alg, vs) in pairs
        ]

        # Cache the FULL validation volumes (one item per run). The trainer
        # runs sliding_window_inference over each volume so the validation
        # metric matches inference-time behavior (overlap + Gaussian blending)
        # instead of scoring independent, zero-padded patches — the latter
        # systematically underestimates F1 by splitting particles at patch
        # boundaries. GPU memory stays bounded by the trainer's sw_batch_size
        # (only a few ROI windows resident at once), not by the volume size.
        val_ds = CacheDataset(data=val_files, transform=augment.get_transforms(), cache_rate=1.0, num_workers=4)

        # batch_size=1: validation is per full volume (volumes differ in shape
        # and cannot be stacked). Throughput is governed by the trainer's
        # sw_batch_size, not this loader. Few workers / no persistent_workers
        # so the cache RAM is reclaimed between infrequent validations.
        val_nw = max(1, train_nw // 4)
        val_loader = DataLoader(
            val_ds, batch_size=1,
            shuffle=False, num_workers=val_nw,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available()
        )

        # Print the data splits
        if self.verbose:
            utils.print_splits(self.myRunIDs, train_files, val_files)

        return train_loader, val_loader

    def get_dataloader_parameters(self):
        """
        Get the datamodule parameters.
        """
        return utils.get_parameters(self) 