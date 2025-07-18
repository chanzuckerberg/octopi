"""
File I/O utilities for YAML and JSON operations.
"""

import os, json, yaml


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
    

def save_results_to_json(results, filename: str):
    """
    Save training results to a JSON file.
    """
    results = prepare_inline_results_json(results)
    with open(os.path.join(filename), "w") as json_file:
        json.dump( results, json_file, indent=4 )
    print(f"Training Results saved to {filename}")


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
    
    parameters = {
        'model': model.get_model_parameters(),
        'optimizer': get_optimizer_parameters(trainer),
        'dataloader': dataloader.get_dataloader_parameters()
    }

    save_parameters_yaml(parameters, filename)
    print(f"Training Parameters saved to {filename}") 

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