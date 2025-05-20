from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, 
    NormalizeIntensityd,
    EnsureChannelFirstd,  
)
from sklearn.model_selection import train_test_split
import copick, torch, os, json, random
from collections import defaultdict
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
        tomogram = get_tomogram_array(run, voxel_spacing, tomo_algorithm)
        segmentation = get_segmentation_array(run, 
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
        tomogram = get_tomogram_array(run, voxel_spacing, tomo_algorithm)
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

##############################################################################################################################

def get_tomogram_array(run, 
                       voxel_size: float = 10, 
                       tomo_type: str = 'wbp',
                       raise_error: bool = True):
    
    voxel_spacing_obj = run.get_voxel_spacing(voxel_size)

    if voxel_spacing_obj is None:
        # Query Avaiable Voxel Spacings
        availableVoxelSpacings = [tomo.voxel_size for tomo in run.voxel_spacings]

        # Report to the user which voxel spacings they can use 
        message = (f"\n[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {tomo_type}"
                   f"\nAvailable spacings are: {', '.join(map(str, availableVoxelSpacings))}\n" ) 
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None
    
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        # Get available algorithms
        availableAlgorithms = [tomo.tomo_type for tomo in run.get_voxel_spacing(voxel_size).tomograms]
        
        # Report to the user which algorithms are available
        message = (f"\n[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {tomo_type}"
                   f"\nAvailable algorithms are: {', '.join(availableAlgorithms)}\n")
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None

    return tomogram.numpy().astype(np.float32)

##############################################################################################################################

def get_segmentation_array(run, 
                           voxel_spacing: float,
                           segmentation_name: str, 
                           session_id=None,
                           user_id=None,
                           raise_error: bool = True):

    seg = run.get_segmentations(name=segmentation_name, 
                                session_id = session_id,
                                user_id = user_id,
                                voxel_size=float(voxel_spacing))
    
    # No Segmentations Are Available, Result in Error
    if len(seg) == 0:
        # Get all available segmentations with their metadata
        available_segs = run.get_segmentations(voxel_size=voxel_spacing)
        seg_info = [(s.name, s.user_id, s.session_id) for s in available_segs]
        
        # Format the information for display
        seg_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                      for name, uid, sid in seg_info]
        
        message = ( f'\nNo segmentation found matching:\n'
                    f'  name: {segmentation_name}, user_id: {user_id}, session_id: {session_id}\n'
                    f'Available segmentations in {run.name} are:\n  ' + 
                    '\n  '.join(seg_details) )
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None

    # No Segmentations Are Available, Result in Error
    if len(seg) > 1:
        print(f'[Warning] More Than 1 Segmentation is Available for the Query Information. '
              f'Available Segmentations are: {seg} '
              f'Defaulting to Loading: {seg[0]}\n')
    seg = seg[0]

    return seg.numpy().astype(np.int8)

##############################################################################################################################

def get_copick_coordinates(run,                    # CoPick run object containing the segmentation data
                           name: str,              # Name of the object or protein for which coordinates are being extracted
                           user_id: str,           # Identifier of the user that generated the picks
                           session_id: str = None, # Identifier of the session that generated the picks
                           voxel_size: float = 10,  # Voxel size of the tomogram, used for scaling the coordinates
                           raise_error: bool = True):
                           
    # Retrieve the pick points associated with the specified object and user ID
    picks = run.get_picks(object_name=name, user_id=user_id, session_id=session_id)

    if len(picks) == 0:
        # Get all available segmentations with their metadata

        available_picks = run.get_picks()
        picks_info = [(s.pickable_object_name, s.user_id, s.session_id) for s in available_picks]
        
        # Format the information for display
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                      for name, uid, sid in picks_info]
        
        message = ( f'\nNo picks found matching:\n'
                    f'  name: {name}, user_id: {user_id}, session_id: {session_id}\n'
                    f'Available picks are:\n  '  
                    + '\n  '.join(picks_details) )
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None
    elif len(picks) > 1:
        # Format pick information for display
        picks_info = [(p.pickable_object_name, p.user_id, p.session_id) for p in picks]
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                        for name, uid, sid in picks_info]

        print(f'[Warning] More than 1 pick is available for the query information.'
              f'\nAvailable picks are:\n  ' + 
              '\n  '.join(picks_details) +
              f'\nDefaulting to loading:\n {picks[0]}\n')
    points = picks[0].points

    # Initialize an array to store the coordinates
    nPoints = len(picks[0].points)                      # Number of points retrieved
    coordinates = np.zeros([len(picks[0].points), 3])   # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(nPoints):
        coordinates[ii,] = [points[ii].location.z / voxel_size,   # Scale z-coordinate by voxel size
                            points[ii].location.y / voxel_size,   # Scale y-coordinate by voxel size
                            points[ii].location.x / voxel_size]   # Scale x-coordinate by voxel size

    # Return the array of coordinates
    return coordinates
    

##############################################################################################################################

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

# def save_parameters_to_json(model, trainer, dataloader, filename: str):

#     parameters = {
#         'model': model.get_model_parameters(),
#         'optimizer': get_optimizer_parameters(trainer),
#         'dataloader': dataloader.get_dataloader_parameters()
#     }
#     parameters = prepare_for_inline_json(parameters)

#     with open(os.path.join(filename), "w") as json_file:
#         json.dump( parameters, json_file, indent=4 )
#     print(f"Training Parameters saved to {filename}")

# def split_datasets(runIDs, 
#                    train_ratio: float = 0.7, 
#                    val_ratio: float = 0.15, 
#                    test_ratio: float = 0.15, 
#                    return_test_dataset: bool = True,
#                    random_state: int = 42):
#     """
#     Splits a given dataset into three subsets: training, validation, and testing. The proportions
#     of each subset are determined by the provided ratios, ensuring that they add up to 1. The
#     function uses a fixed random state for reproducibility.

#     Parameters:
#     - runIDs: The complete dataset that needs to be split.
#     - train_ratio: The proportion of the dataset to be used for training.
#     - val_ratio: The proportion of the dataset to be used for validation.
#     - test_ratio: The proportion of the dataset to be used for testing.

#     Returns:
#     - trainRunIDs: The subset of the dataset used for training.
#     - valRunIDs: The subset of the dataset used for validation.
#     - testRunIDs: The subset of the dataset used for testing.
#     """

#     # Ensure the ratios add up to 1
#     assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must add up to 1.0"

#     # First, split into train and remaining (30%)
#     trainRunIDs, valRunIDs = train_test_split(runIDs, test_size=(1 - train_ratio), random_state=random_state)

#     # (Optional) split the remaining into validation and test
#     if return_test_dataset: 
#         valRunIDs, testRunIDs = train_test_split(
#             valRunIDs,
#             test_size=(test_ratio / (val_ratio + test_ratio)),
#             random_state=random_state,
#         )
#     else:
#         testRunIDs = None

#     return trainRunIDs, valRunIDs, testRunIDs
