from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, 
    NormalizeIntensityd,
    EnsureChannelFirstd,  
)
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import copick, torch, os, json
from tqdm import tqdm

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
    return test_loader

##############################################################################################################################

def get_tomogram_array(run, 
                       voxel_size: float = 10, 
                       tomo_type: str = 'wbp'):
    
    voxel_spacing_obj = run.get_voxel_spacing(voxel_size)

    if voxel_spacing_obj is None:
        # Query Avaiable Voxel Spacings
        availableVoxelSpacings = [tomo.voxel_size for tomo in run.voxel_spacings]

        # Report to the user which spacings they can use 
        raise ValueError(f"Voxel Spacings of '{voxel_size}' was not found. "
                         f"Available spacings are: {', '.join(map(str, availableVoxelSpacings))}")
    
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        # Get available algorithms
        availableAlgorithms = [tomo.tomo_type for tomo in run.get_voxel_spacing(voxel_size).tomograms]
        
        # Report to the user
        raise ValueError(f"The tomogram with the algorithm '{tomo_type}' was not found. "
                         f"Available algorithms are: {', '.join(availableAlgorithms)}")


    return tomogram.numpy()

##############################################################################################################################

def get_segmentation_array(run, 
                           voxel_spacing: float,
                           segmentation_name: str, 
                           session_id=None,
                           user_id=None,
                           is_multilabel=True):

    seg = run.get_segmentations(name=segmentation_name, 
                                       session_id = session_id,
                                       user_id = user_id,
                                       voxel_size=voxel_spacing, 
                                       is_multilabel=is_multilabel)
    
    # No Segmentations Are Available, Result in Error
    if len(seg) == 0:
        raise ValueError(f'Missing Segmentation for Name: {segmentation_name}, UserID: {user_id}, SessionID: {session_id}')

    # No Segmentations Are Available, Result in Error
    if len(seg) > 1:
        print(f'[Warning] More Than 1 Segmentation is Available for the Query Information. '
              f'Available Segmentations are: {seg} '
              f'Defaulting to Loading: {seg[0]}\n')
    seg = seg[0]

    return seg.numpy()

##############################################################################################################################

def get_num_classes(copick_config_path: str):

    root = copick.from_file(copick_config_path)
    return len(root.pickable_objects) + 1

##############################################################################################################################

def split_datasets(runIDs, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, 
                   return_test_dataset: bool = True,
                   random_state: int = 42):
    """
    Splits a given dataset into three subsets: training, validation, and testing. The proportions
    of each subset are determined by the provided ratios, ensuring that they add up to 1. The
    function uses a fixed random state for reproducibility.

    Parameters:
    - runIDs: The complete dataset that needs to be split.
    - train_ratio: The proportion of the dataset to be used for training.
    - val_ratio: The proportion of the dataset to be used for validation.
    - test_ratio: The proportion of the dataset to be used for testing.

    Returns:
    - trainRunIDs: The subset of the dataset used for training.
    - valRunIDs: The subset of the dataset used for validation.
    - testRunIDs: The subset of the dataset used for testing.
    """

    # Ensure the ratios add up to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must add up to 1.0"

    # First, split into train and remaining (30%)
    trainRunIDs, valRunIDs = train_test_split(runIDs, test_size=(1 - train_ratio), random_state=random_state)

    # (Optional) split the remaining into validation and test
    if return_test_dataset: 
        valRunIDs, testRunIDs = train_test_split(
            valRunIDs,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_state,
        )
    else:
        testRunIDs = None

    return trainRunIDs, valRunIDs, testRunIDs

##############################################################################################################################

def load_copick_config(path: str):

    if os.path.isfile(path):
        root = copick.from_file(path)
    else:
        raise ValueError(f'{path} is not a valid path!')
    
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

def get_model_parameters(model):

    model_parameters = {
        'model_name': model.__class__.__name__, 
        'channels': model.channels, 
        'strides': model.strides,
        'res_units': model.num_res_units        
    }

    return model_parameters

def get_optimizer_parameters(trainer):

    optimizer_parameters = {
        'optimizer': trainer.optimizer.__class__.__name__,
        'lr': trainer.optimizer.param_groups[0]['lr'],
        'loss_function': trainer.loss_function.__class__.__name__,
        'metrics_function': trainer.metrics_function.__class__.__name__,
        'my_num_samples': trainer.num_samples,
        'reload_frequency': trainer.reload_frequency        
    }

    return optimizer_parameters

def save_parameters_to_json(model, trainer, dataloader, filename: str):

    parameters = {
        'model': get_model_parameters(model),
        'optimizer': get_optimizer_parameters(trainer),
        'dataloader': dataloader.get_dataloader_parameters()
    }
    parameters = prepare_for_inline_json(parameters)

    with open(os.path.join(filename), "w") as json_file:
        json.dump( parameters, json_file, indent=4 )
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