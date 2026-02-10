"""
Configuration utilities for MLflow setup and reproducibility.
"""

from dotenv import load_dotenv
import torch, numpy as np
import os, random
import octopi


def mlflow_setup():
    """
    Set up MLflow configuration from environment variables.
    """
    module_root = os.path.dirname(octopi.__file__)
    dotenv_path = module_root.replace('src/octopi','') + '.env'
    load_dotenv(dotenv_path=dotenv_path)

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

    return os.getenv('MLFLOW_TRACKING_URI')


def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
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