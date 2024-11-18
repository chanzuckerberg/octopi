from dotenv import load_dotenv
import torch, random, os
from typing import List
import numpy as np

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