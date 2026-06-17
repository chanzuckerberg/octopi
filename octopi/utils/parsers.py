"""
Argument parsing and configuration utilities.
"""

from typing import List, Tuple, Union
import argparse

def parse_list(value: str) -> List[str]:
    """
    Parse a string representing a list of items.
    Supports formats like '[item1,item2,item3]' or 'item1,item2,item3'.
    """
    value = value.strip("[]")  # Remove brackets if present
    return [x.strip() for x in value.split(",")]


def parse_int_list(value: str) -> List[int]:
    """
    Parse a string representing a list of integers.
    Supports formats like '[1,2,3]' or '1,2,3'.
    """
    return [int(x) for x in parse_list(value)]


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


def parse_target(value: str) -> Tuple[str, Union[str, None], Union[str, None]]:
    """
    Parse a target / segmentation query into (name, user_id, session_id).

    Accepts the copick URI grammar (preferred):
      - "name"
      - "name:user_id"
      - "name:user_id/session_id"
    and the legacy comma form for backward compatibility:
      - "name,user_id,session_id"

    The voxel size is NOT part of this query (it is taken from --tomo-uri), so a
    trailing '@voxel_size' is rejected.
    """
    # Legacy comma form (unambiguous: ':' / '/' never appear in the comma form).
    if ',' in value:
        parts = value.split(',')
        if len(parts) == 1:
            return parts[0], None, None
        elif len(parts) == 3:
            obj_name, user_id, session_id = parts
            return obj_name, user_id, session_id
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid target format: '{value}'. Expected 'name' or 'name,user_id,session_id'."
            )

    # URI form: name[:user_id[/session_id]]
    if '@' in value:
        raise argparse.ArgumentTypeError(
            f"Invalid query '{value}': do not include '@voxel_size' here; the voxel size "
            f"is derived from --tomo-uri."
        )
    name, sep, rest = value.partition(':')
    if not sep:
        return name, None, None
    user_id, _, session_id = rest.partition('/')
    return name, (user_id or None), (session_id or None)


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

        # if ',' in config_entry:
        #     parts = config_entry.split(',')
        #     if len(parts) == 2:
        #         # Entry with session name and config path
        #         session_name, config_path = parts
        #         copick_configs[session_name] = {"path": config_path, "algorithm": None}
        #     elif len(parts) == 3:
        #         # Entry with session name, config path, and algorithm
        #         session_name, config_path, algorithm = parts
        #         copick_configs[session_name] = {"path": config_path, "algorithm": algorithm}    
        # else:
        #     copick_configs = config_entry

    return copick_configs


def parse_data_split(value: str) -> Tuple[float, float, float]:
    """
    Parse data split ratios from string input.
    
    Args:
        value: Either a single float (e.g., "0.8") or two comma-separated floats (e.g., "0.7,0.1")
    
    Returns:
        Tuple of (train_ratio, val_ratio, test_ratio)
    
    Examples:
        "0.8" -> (0.8, 0.2, 0.0)
        "0.7,0.1" -> (0.7, 0.1, 0.2)
    """
    parts = value.split(',')
    
    if len(parts) == 1:
        # Single value provided - use it as train ratio
        train_ratio = float(parts[0])
        val_ratio = 1.0 - train_ratio
        test_ratio = 0.0
    elif len(parts) == 2:
        # Two values provided - use as train and val ratios
        train_ratio = float(parts[0])
        val_ratio = float(parts[1])
        test_ratio = 1.0 - train_ratio - val_ratio
    else:
        raise ValueError("Data split must be either a single value or two comma-separated values")
    
    # Validate ratios
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("All ratios must be non-negative")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    return round(train_ratio, 2), round(val_ratio, 2), round(test_ratio, 2) 