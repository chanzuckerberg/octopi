from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, 
    NormalizeIntensityd,
    EnsureChannelFirstd,  
)
from sklearn.model_selection import train_test_split
import copick, torch, os, json, random
from collections import defaultdict
from copick_utils.io import readers
from octopi import utils
from typing import List
from tqdm import tqdm
import numpy as np

##############################################################################################################################    

def load_training_data(root, 
                       runIDs: List[str],
                       voxel_spacing: float, 
                       tomo_algorithm: str, 
                       segmenation_name: str,
                       segmentation_session_id: str = None,
                       segmentation_user_id: str = None,
                       progress_update: bool = True):
    
    data_dicts = []
    # Use tqdm for progress tracking only if progress_update is True
    iterable = tqdm(runIDs, desc="Loading Training Data") if progress_update else runIDs
    for runID in iterable:
        run = root.get_run(str(runID))
        tomogram = readers.tomogram(run, voxel_spacing, tomo_algorithm)
        segmentation = readers.segmentation(run, 
                                              voxel_spacing,
                                              segmenation_name,
                                              segmentation_session_id, 
                                              segmentation_user_id)
        data_dicts.append({"image": tomogram, "label": segmentation})

    return data_dicts 

##############################################################################################################################    

def load_predict_data(root, 
                      runIDs: List[str],
                      voxel_spacing: float, 
                      tomo_algorithm: str):
    
    data_dicts = []
    for runID in tqdm(runIDs):
        run = root.get_run(str(runID))
        tomogram = readers.tomogram(run, voxel_spacing, tomo_algorithm)
        data_dicts.append({"image": tomogram})

    return data_dicts 

##############################################################################################################################    

def create_predict_dataloader(
    root,
    voxel_spacing: float, 
    tomo_algorithm: str,       
    runIDs: str = None,       
    ): 

    # define pre transforms
    pre_transforms = Compose(
        [   EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"]),
    ])

    # Split trainRunIDs, validateRunIDs, testRunIDs
    if runIDs is None:
        runIDs = [run.name for run in root.runs]
    test_files = load_predict_data(root, runIDs, voxel_spacing, tomo_algorithm) 

    bs = min( len(test_files), 4)
    test_ds = CacheDataset(data=test_files, transform=pre_transforms)
    test_loader = DataLoader(test_ds, 
                            batch_size=bs, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=torch.cuda.is_available())
    return test_loader, test_ds

def adjust_to_multiple(value, multiple = 16):
    return int((value // multiple) * multiple)

def get_input_dimensions(dataset, crop_size: int):
    nx = dataset[0]['image'].shape[1]
    if crop_size > nx:
        first_dim = adjust_to_multiple(nx/2)
        return first_dim, crop_size, crop_size
    else:
        return crop_size, crop_size, crop_size

def get_num_classes(copick_config_path: str):

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

##############################################################################################################################

def load_copick_config(path: str):

    if os.path.isfile(path):
        root = copick.from_file(path)
    else:
        raise FileNotFoundError(f"Copick Config Path does not exist: {path}")
    
    return root

##############################################################################################################################

# Helper function to flatten and serialize nested parameters
def flatten_params(params, parent_key=''):
    flattened = {}
    for key, value in params.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_params(value, new_key))
        elif isinstance(value, list):
            flattened[new_key] = ', '.join(map(str, value))  # Convert list to a comma-separated string
        else:
            flattened[new_key] = value
    return flattened

# Manually join specific lists into strings for inline display
def prepare_for_inline_json(data):
    for key in ["trainRunIDs", "valRunIDs", "testRunIDs"]:
        if key in data['dataloader']:
            data['dataloader'][key] = f"[{', '.join(map(repr, data['dataloader'][key]))}]"

    for key in ['channels', 'strides']:
        if key in data['model']:
                data['model'][key] = f"[{', '.join(map(repr, data['model'][key]))}]"
    return data

def get_optimizer_parameters(trainer):

    optimizer_parameters = {
        'my_num_samples': trainer.num_samples,  
        'val_interval': trainer.val_interval,
        'lr': trainer.optimizer.param_groups[0]['lr'],
        'optimizer': trainer.optimizer.__class__.__name__,
        'metrics_function': trainer.metrics_function.__class__.__name__,
        'loss_function': trainer.loss_function.__class__.__name__,
    }

    # Log Tversky Loss Parameters
    if trainer.loss_function.__class__.__name__ == 'TverskyLoss':
        optimizer_parameters['alpha'] = trainer.loss_function.alpha
    elif trainer.loss_function.__class__.__name__ == 'FocalLoss':
        optimizer_parameters['gamma'] = trainer.loss_function.gamma
    elif trainer.loss_function.__class__.__name__ == 'WeightedFocalTverskyLoss':
        optimizer_parameters['alpha'] = trainer.loss_function.alpha
        optimizer_parameters['gamma'] = trainer.loss_function.gamma
        optimizer_parameters['weight_tversky'] = trainer.loss_function.weight_tversky
    elif trainer.loss_function.__class__.__name__ == 'FocalTverskyLoss':
        optimizer_parameters['alpha'] = trainer.loss_function.alpha
        optimizer_parameters['gamma'] = trainer.loss_function.gamma

    return optimizer_parameters

def save_parameters_to_yaml(model, trainer, dataloader, filename: str):

    parameters = {
        'model': model.get_model_parameters(),
        'optimizer': get_optimizer_parameters(trainer),
        'dataloader': dataloader.get_dataloader_parameters()
    }

    utils.save_parameters_yaml(parameters, filename)
    print(f"Training Parameters saved to {filename}")

def prepare_inline_results_json(results):
    # Traverse the dictionary and format lists of lists as inline JSON
    for key, value in results.items():
        # Check if the value is a list of lists (like [[epoch, value], ...])
        if isinstance(value, list) and all(isinstance(item, list) and len(item) == 2 for item in value):
            # Format the list of lists as a single-line JSON string
            results[key] = json.dumps(value)
    return results    

# Check to See if I'm Happy with This... Maybe Save as H5 File? 
def save_results_to_json(results, filename: str):

    results = prepare_inline_results_json(results)
    with open(os.path.join(filename), "w") as json_file:
        json.dump( results, json_file, indent=4 )
    print(f"Training Results saved to {filename}")

##############################################################################################################################

