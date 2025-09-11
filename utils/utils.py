import os
import json
import torch
import numpy as np
import random

def inherit_docstring(cls):
    """
    Inherit docstrings from base classes to methods without docstrings.
    
    This decorator automatically applies the docstring from a base class method 
    to a derived class method if the derived method doesn't have its own docstring.
    
    Args:
        cls: The class to enhance with inherited docstrings
        
    Returns:
        cls: The enhanced class
    """
    for name, func in vars(cls).items():
        if not callable(func) or func.__doc__ is not None:
            continue
            
        # Look for same method in base classes
        for base in cls.__bases__:
            base_func = getattr(base, name, None)
            if base_func and getattr(base_func, "__doc__", None):
                func.__doc__ = base_func.__doc__
                break
                
    return cls


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


def load_json_as_list(input_file: str) -> list:
    """Load a JSON file as a list of dictionaries."""
    res = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        res.append(d)
    return res


def create_directory_for_file(file_path) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed((seed + 3) % 2**32)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)