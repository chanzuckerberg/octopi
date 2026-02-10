"""
Data loading, processing, and dataset operations for the datasets module.
"""

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, 
    NormalizeIntensityd,
    EnsureChannelFirstd,  
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from octopi.datasets.loader import LoadCopickPredictd
from sklearn.model_selection import train_test_split
from copick.util.uri import resolve_copick_objects
import copick, torch, os, random, threading
from collections import defaultdict
from typing import List, Optional
from tqdm import tqdm

_thread_state = threading.local()

def create_predict_dataloader(
    config,
    voxel_size: float, 
    tomo_alg: str,       
    runIDs: str = None,
    batch_size: int = 1,
    nworkers: int = None,
    ): 
    """
    Create a dataloader for prediction data.
    """
    # define pre transforms
    xforms = Compose([   
        LoadCopickPredictd(),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"]),
    ])

    # Prepare the list of runIDs if none are provide
    if runIDs is None:
        runIDs = [run.name for run in root.runs]

    # Validate RunIDs before creating the dataloader
    uri = f'{tomo_alg}@{voxel_size}'
    runIDs = _check_valid_runs(config, uri, runIDs)
    test_files = [{
        'run': runid, 'root': config,
        'vol_uri': uri
        } for runid in runIDs
    ]
    test_ds = Dataset(data=test_files, transform=xforms)

    if nworkers is None:
        nworkers = min(2, torch.get_num_threads())
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, shuffle=False, 
        num_workers=nworkers, 
        pin_memory=torch.cuda.is_available()
    )
    return test_loader, test_ds

def _get_root(config: str):
    # One root per thread
    if not hasattr(_thread_state, "root"):
        _thread_state.root = copick.from_file(config)
    return _thread_state.root

def _check_one_run(config: str, vol_uri: str, runID: str) -> Optional[str]:
    root = _get_root(config)
    vol = resolve_copick_objects(vol_uri, root, "tomogram", run_name=runID)
    return runID if len(vol) > 0 else None

def _check_valid_runs(config: str, vol_uri: str, runIDs: List[str], max_workers: int = 16) -> List[str]:

    valid: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_one_run, config, vol_uri, runID) for runID in runIDs]
        for fut in as_completed(futures):
            runID = fut.result()
            if runID is not None:
                valid.append(runID)

    # Optional: keep original order
    valid_set = set(valid)
    valid = [rid for rid in runIDs if rid in valid_set]

    if not valid:
        raise ValueError(f"No valid runs found for given volume URI: {vol_uri}")

    return valid

def adjust_to_multiple(value, multiple = 16):
    """
    Adjust a value to be a multiple of the specified number.
    """
    return int((value // multiple) * multiple)


def get_input_dimensions(dataset, crop_size: int):
    """
    Get input dimensions for the dataset.
    """
    nx = dataset[0]['image'].shape[1]
    if crop_size > nx:
        first_dim = adjust_to_multiple(nx/2)
        return first_dim, crop_size, crop_size
    else:
        return crop_size, crop_size, crop_size


def get_num_classes(copick_config_path: str):
    """
    Get the number of classes from a CoPick configuration.
    """
    root = copick.from_file(copick_config_path)
    return len(root.pickable_objects) + 1


def split_multiclass_dataset(runIDs, 
                             train_ratio: float = 0.7, 
                             val_ratio: float = 0.15, 
                             test_ratio: float = 0.15, 
                             return_test_dataset: bool = True,
                             random_state: int = 42):
    """
    Splits a given dataset into three subsets: training, validation, and testing. If the dataset
    has categories (as tuples), splits are balanced across all categories. If the dataset is a 1D
    list, it is split without categorization.

    Parameters:
    - runIDs: A list of items to split. It can be a 1D list or a list of tuples (category, value).
    - train_ratio: Proportion of the dataset for training.
    - val_ratio: Proportion of the dataset for validation.
    - test_ratio: Proportion of the dataset for testing.
    - return_test_dataset: Whether to return the test dataset.
    - random_state: Random state for reproducibility.

    Returns:
    - trainRunIDs: Training subset.
    - valRunIDs: Validation subset.
    - testRunIDs: Testing subset (if return_test_dataset is True, otherwise None).
    """

    # Ensure the ratios add up to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must add up to 1.0"

    # Check if the dataset has categories
    if isinstance(runIDs[0], tuple) and len(runIDs[0]) == 2:
        # Group by category
        grouped = defaultdict(list)
        for item in runIDs:
            grouped[item[0]].append(item)

        # Split each category
        trainRunIDs, valRunIDs, testRunIDs = [], [], []
        for category, items in grouped.items():
            # Shuffle for randomness
            random.shuffle(items)
            # Split into train and remaining
            train_items, remaining = train_test_split(items, test_size=(1 - train_ratio), random_state=random_state)
            trainRunIDs.extend(train_items)

            if return_test_dataset:
                # Split remaining into validation and test
                val_items, test_items = train_test_split(
                    remaining,
                    test_size=(test_ratio / (val_ratio + test_ratio)),
                    random_state=random_state,
                )
                valRunIDs.extend(val_items)
                testRunIDs.extend(test_items)
            else:
                valRunIDs.extend(remaining)
                testRunIDs = []
    else:
        # If no categories, split as a 1D list
        trainRunIDs, remaining = train_test_split(runIDs, test_size=(1 - train_ratio), random_state=random_state)
        if return_test_dataset:
            valRunIDs, testRunIDs = train_test_split(
                remaining,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                random_state=random_state,
            )
        else:
            valRunIDs = remaining
            testRunIDs = []

    return trainRunIDs, valRunIDs, testRunIDs    


def load_copick_config(path: str):
    """
    Load a CoPick configuration from file.
    """
    if os.path.isfile(path):
        root = copick.from_file(path)
    else:
        raise FileNotFoundError(f"Copick Config Path does not exist: {path}")
    
    return root 