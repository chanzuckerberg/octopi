"""
File I/O utilities for YAML and JSON operations.
"""

import os, json, yaml, copick, glob
import pandas as pd


# Create a custom dumper that uses flow style for lists only.
class InlineListDumper(yaml.SafeDumper):
    def represent_list(self, data):
        node = super().represent_list(data)
        node.flow_style = True  # Use inline style for lists
        return node


def save_parameters_yaml(params: dict, output_path: str):
    """
    Save parameters to a YAML file.
    """
    InlineListDumper.add_representer(list, InlineListDumper.represent_list)
    with open(output_path, 'w') as f:
        yaml.dump(params, f, Dumper=InlineListDumper, default_flow_style=False, sort_keys=False)


def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return the contents as a dictionary.
    """
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"File not found: {path}")

def save_results_to_csv(results, filename: str):
    """Save training results to a CSV file (aligned to validation steps)."""
    data = {}
    
    # Get validation steps (these are the steps we want to keep)
    val_steps = set()
    for key, value in results.items():
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            if key.startswith(('val_', 'avg_', 'f1_', 'recall_', 'precision_', 'fbeta_')):
                val_steps.update([item[0] for item in value])
                break
    
    val_steps = sorted(val_steps)
    data['step'] = val_steps
    
    # Extract all metrics, filtering to validation steps only
    for key, value in results.items():
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            step_to_value = {item[0]: item[1] for item in value}
            data[key] = [step_to_value.get(step, None) for step in val_steps]
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"📊 Training Results saved to {filename}")

def prepare_inline_results_json(results):
    """
    Prepare results for inline JSON formatting.
    """
    # Traverse the dictionary and format lists of lists as inline JSON
    for key, value in results.items():
        # Check if the value is a list of lists (like [[epoch, value], ...])
        if isinstance(value, list) and all(isinstance(item, list) and len(item) == 2 for item in value):
            # Format the list of lists as a single-line JSON string
            results[key] = json.dumps(value)
    return results 

def get_optimizer_parameters(trainer):
    """
    Extract optimizer parameters from a trainer object.
    """
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
    """
    Save training parameters to a YAML file.
    """

    # Check for the target configuration file for model labels
    target_config = check_target_config_path(dataloader)
    
    # Extract and flatten parameters
    parameters = {
        'model': model.get_model_parameters(),
        'labels': target_config['input']['labels'],
        'optimizer': get_optimizer_parameters(trainer),
        'dataloader': dataloader.get_dataloader_parameters()
    }

    save_parameters_yaml(parameters, filename)
    print(f"⚙️ Training Parameters saved to {filename}") 

def flatten_params(params, parent_key=''):
    """
    Helper function to flatten and serialize nested parameters.
    """
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


def prepare_for_inline_json(data):
    """
    Manually join specific lists into strings for inline display.
    """
    for key in ["trainRunIDs", "valRunIDs", "testRunIDs"]:
        if key in data['dataloader']:
            data['dataloader'][key] = f"[{', '.join(map(repr, data['dataloader'][key]))}]"

    for key in ['channels', 'strides']:
        if key in data['model']:
                data['model'][key] = f"[{', '.join(map(repr, data['model'][key]))}]"
    return data

def check_target_config_path(data_generator):
    """
    Check for the target configuration file in the CoPick overlay or static root directories.
    If the session_id is not provided, search for the most recent file matching the target name
    """
    
    # Open the Copick Project for MultiConfig or SingleConfig Workflow
    if isinstance(data_generator.config, dict):
        sessions = list(data_generator.config.keys())
        config_path = data_generator.config[sessions[0]]
    else:
        config_path = data_generator.config

    # Get the Target Config File
    return get_config(
        config_path, data_generator.target_name, 'targets',
        data_generator.target_user_id, data_generator.target_session_id
    )

def get_config(config_path, name, process, user_id=None, session_id=None):
    """
    Get the configuration for a specific process and target name.
    """

    # Get the Overlay and Static Roots
    root = copick.from_file(config_path)

    # Remove the local:// prefix from static_root if it exists  
    overlay_root = remove_prefix(root.config.overlay_root)
    try:  # Check if static_root is available
        static_root = remove_prefix(root.config.static_root)
    except: # If not, set it to None
        static_root = None
    
    # Two Search Patterns, Either only a name provided or name, user_id, session_id
    if session_id is None:
        pattern = glob.glob(os.path.join(overlay_root, 'logs', f"{process}_*{name}.yaml"))
        if len(pattern) == 0 and static_root is not None:
            pattern = glob.glob(os.path.join(static_root, 'logs', f"{process}_*{name}.yaml"))
        fname = pattern[-1]
    else:
        fname = f"{process}-{user_id}_{session_id}_{name}.yaml"

    # The Target Config File Should Either in be the Overlay or Static Root
    if os.path.exists(os.path.join(overlay_root, 'logs', fname)):
        path = os.path.join(overlay_root, 'logs', fname)
    elif static_root is not None and os.path.exists(os.path.join(static_root, 'logs', fname)):
        path = os.path.join(static_root, 'logs', fname)
    else:
        raise FileNotFoundError(f"Target config file not found: {fname}")

    # Load the Target Config File
    with open(path, 'r') as f:
        target_config = yaml.safe_load(f)
    return target_config

def remove_prefix(text: str) -> str:
    """
    Remove a prefix from a string if it exists.
    """
    # Check if the text is None
    if text is None:
        return None
    elif text[:8] == 'local://':
        text = text[8:]
    return text