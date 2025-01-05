from monai.networks.nets import UNet, AttentionUnet
from typing import List, Tuple, Union
from dotenv import load_dotenv
import torch, random, os
from typing import List
import numpy as np
import argparse

##############################################################################################################################

def mlflow_setup():

    # MLflow setup
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if not password or not username:
        print("Password not found in environment, loading from .env file...")
        load_dotenv()  # Loads environment variables from a .env file
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    
    # Check again after loading .env file
    if not password:
        raise ValueError("Password is not set in environment variables or .env file!")
    else:
        print("Password loaded successfully")
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password    

##############################################################################################################################

def set_seed(seed):
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure reproducibility of operations by disabling certain optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################################################################################    

def get_copick_coordinates(copick_run,             # CoPick run object containing the segmentation data
                           name: str,              # Name of the object or protein for which coordinates are being extracted
                           user_id: str,           # Identifier of the user that generated the picks
                           session_id: str = None, # Identifier of the session that generated the picks
                           voxel_size: float = 10  # Voxel size of the tomogram, used for scaling the coordinates
                           ):
                           
    # Retrieve the pick points associated with the specified object and user ID
    points = copick_run.get_picks(object_name=name, user_id=user_id, session_id=session_id)[0].points
    
    # Initialize an array to store the coordinates
    nPoints = len(points)                      # Number of points retrieved
    coordinates = np.zeros([len(points), 3])   # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(nPoints):
        coordinates[ii,] = [points[ii].location.z / voxel_size,   # Scale z-coordinate by voxel size
                            points[ii].location.y / voxel_size,   # Scale y-coordinate by voxel size
                            points[ii].location.x / voxel_size]   # Scale x-coordinate by voxel size
    
    # Return the array of coordinates
    return coordinates

###############################################################################################################################

def parse_list(value: str) -> List[str]:
    """
    Parse a string representing a list of items.
    Supports formats like '[item1,item2,item3]' or 'item1,item2,item3'.
    """
    value = value.strip("[]")  # Remove brackets if present
    return [x.strip() for x in value.split(",")]

###############################################################################################################################

def parse_int_list(value: str) -> List[int]:
    """
    Parse a string representing a list of integers.
    Supports formats like '[1,2,3]' or '1,2,3'.
    """
    return [int(x) for x in parse_list(value)]

###############################################################################################################################

def string2bool(value: str):
    """
    Custom function to convert string values to boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'True', 'true', 't', '1', 'yes'}:
        return True
    elif value.lower() in {'False', 'false', 'f', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

###############################################################################################################################

def parse_target(value: str) -> Tuple[str, Union[str, None], Union[str, None]]:
    """
    Parse a single target string.
    Expected formats:
      - "name"
      - "name,user_id,session_id"
    """
    parts = value.split(',')
    if len(parts) == 1:
        obj_name = parts[0]
        return obj_name, None, None
    elif len(parts) == 3:
        obj_name, user_id, session_id = parts
        return obj_name, user_id, session_id
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid target format: '{value}'. Expected 'name' or 'name,user_id,session_id'."
        )


def parse_seg_target(value: str) -> List[Tuple[str, Union[str, None], Union[str, None]]]:
    """
    Parse segmentation targets. Each target can have the format:
      - "name"
      - "name,user_id,session_id"
    Multiple targets can be comma-separated.
    """
    targets = []
    for target in value.split(';'):  # Use ';' as a separator for multiple targets
        parts = target.split(',')
        if len(parts) == 1:
            name = parts[0]
            targets.append((name, None, None))
        elif len(parts) == 3:
            name, user_id, session_id = parts
            targets.append((name, user_id, session_id))
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid seg-target format: '{target}'. Expected 'name' or 'name,user_id,session_id'."
            )
    return targets

def parse_copick_configs(config_entries: List[str]):
    """
    Parse a string representing a list of CoPick configuration file paths.
    """
    # Process the --config arguments into a dictionary
    copick_configs = {}

    # import pdb; pdb.set_trace()

    for config_entry in config_entries:
        if ',' in config_entry:
            # Entry has a session name and a config path
            try:
                session_name, config_path = config_entry.split(',', 1)
                copick_configs[session_name] = config_path
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid format for --config entry: '{config_entry}'. Expected 'session_name,/path/to/config.json'."
                )
        else:
            # Single configuration path without a session name
            # if "default" in copick_configs:
            #     raise argparse.ArgumentTypeError(
            #         f"Only one single-path --config entry is allowed when using default configurations. "
            #         f"Detected duplicate: {config_entry}"
            #     )
            # copick_configs["default"] = config_entry
            copick_configs = config_entry

    return copick_configs

##############################################################################################################################

def create_model(model_type, n_classes, channels, strides_pattern, num_res_units, device):
    """
    Create either a UNet or AttentionUnet model based on trial parameters.
    
    Args:
        trial: Optuna trial object
        n_classes: Number of output classes
        channels: List of channel sizes
        strides_pattern: List of stride values
        num_res_units: Number of residual units (only used for UNet)
        device: torch device to place model on
    """
    
    if model_type == "Unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=channels,
            strides=strides_pattern,
            num_res_units=num_res_units,
        )
    elif model_type == "AttentionUnet":  # AttentionUnet
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=channels,
            strides=strides_pattern,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}, Available models are: 'Unet', 'AttentionUnet'")
    
    return model.to(device)