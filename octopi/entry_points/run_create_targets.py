import octopi.processing.create_targets_from_picks as create_targets
from typing import List, Tuple, Union
from octopi.utils import io, parsers
from collections import defaultdict
import argparse, copick, yaml, os
from tqdm import tqdm
import numpy as np

def create_sub_train_targets(
    config: str,
    pick_targets: List[Tuple[str, Union[str, None], Union[str, None]]],  # Updated type without radius
    seg_targets: List[Tuple[str, Union[str, None], Union[str, None]]],
    voxel_size: float,
    radius_scale: float,    
    tomogram_algorithm: str,
    target_segmentation_name: str,
    target_user_id: str,
    target_session_id: str,
    run_ids: List[str],    
    ):

    # Load Copick Project 
    root = copick.from_file(config)

    # Create empty dictionary for all targets
    train_targets = defaultdict(dict)

    # Create dictionary for particle targets
    for t in pick_targets:
        # Parse the target
        obj_name, user_id, session_id = t 
        obj = root.get_object(obj_name)

        # Check if the object is valid
        if obj is None:
            print(f'Warning - Skipping Particle Target: "{obj_name}", as it is not a valid name in the config file.')
            continue
        
        # Get the label and radius of the object
        label = obj.label
        info = {
            "label": label,
            "user_id": user_id,
            "session_id": session_id,
            "is_particle_target": True,
            "radius": root.get_object(obj_name).radius,
        }
        train_targets[obj_name] = info

    # Create dictionary for segmentation targets
    train_targets = add_segmentation_targets(root, seg_targets, train_targets)

    create_targets.generate_targets(
        root, train_targets, voxel_size, tomogram_algorithm, radius_scale,
        target_segmentation_name, target_user_id, 
        target_session_id, run_ids
    )


def create_all_train_targets(
    config: str,
    seg_targets: List[List[Tuple[str, Union[str, None], Union[str, None]]]],
    picks_session_id: str,
    picks_user_id: str,
    voxel_size: float,
    radius_scale: float,
    tomogram_algorithm: str,
    target_segmentation_name: str,
    target_user_id: str,
    target_session_id: str,
    run_ids: List[str],    
    ):     

    # Load Copick Project 
    root = copick.from_file(config)

    # Create empty dictionary for all targets
    target_objects = defaultdict(dict)

    # Create dictionary for particle targets
    for object in root.pickable_objects:
        info = {
            "label": object.label,
            "radius": object.radius,
            "user_id": picks_user_id,
            "session_id": picks_session_id,
            "is_particle_target": True,
        }
        target_objects[object.name] = info

    # Create dictionary for segmentation targets
    target_objects = add_segmentation_targets(root, seg_targets, target_objects)

    create_targets.generate_targets(
        root, target_objects, voxel_size, tomogram_algorithm, 
        radius_scale, target_segmentation_name, target_user_id, 
        target_session_id, run_ids 
    )

def add_segmentation_targets(
    root,
    seg_targets,
    train_targets: dict,
    ):

    # Create dictionary for segmentation targets
    for s in seg_targets:

        # Parse Segmentation Target
        obj_name, user_id, session_id = s

        # Add Segmentation Target
        try:
            info = {
                "label": root.get_object(obj_name).label,
                "user_id": user_id,
                "session_id": session_id,
                "is_particle_target": False,                 
                "radius": None,    
            }
            train_targets[obj_name] = info

        # If Segmentation Target is not found, print warning
        except:
            print(f'Warning - Skipping Segmentation Name: "{obj_name}", as it is not a valid object in the Copick project.')

    return train_targets    

def parse_args():
    """
    Parse command-line arguments for generating segmentation targets from CoPick configurations.

    This tool allows researchers to specify protein labels for training in two ways:
    
    1. **Manual Specification:** Users can define a subset of pickable objects from the CoPick configuration file.
       - Specify a target protein using `--target name`, or refine selection with `--target name,user_id,session_id`.
       - This enables flexible training target customization from multiple sources.

    2. **Automated Query:** Instead of specifying targets explicitly, users can provide a session ID (`--picks-session-id`) and/or 
       user ID (`--picks-user-id`). DeepFindET will automatically retrieve all pickable objects associated with the query.
    
    The tool also allows customization of tomogram reconstruction settings and segmentation parameters.

    Example Usage:
        - Manual Specification:
            ```bash
            python create_targets.py --config config.json --target ribosome --target apoferritin,123,456
            ```
        - Automated Query:
            ```bash
            python create_targets.py --config config.json --picks-session-id 123 --picks-user-id 456
            ```

    Output segmentation data is saved in a structured YAML file.
    """
    parser = argparse.ArgumentParser(
        description=f"""Generate segmentation targets from CoPick configurations with either --target flag (which lets users specify a subset of pickable objects) or --picks-session-id and --picks-user-id flags (which lets users specify a sessionID and userID to automatically retrieve all pickable objects associated with the query).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    input_group.add_argument("--target", type=parsers.parse_target, action="append", default=None, help='Target specifications: "name" or "name,user_id,session_id".')
    input_group.add_argument("--picks-session-id", type=str, default=None, help="Session ID for the picks.")
    input_group.add_argument("--picks-user-id", type=str, default=None, help="User ID associated with the picks.")
    input_group.add_argument("--seg-target", type=parsers.parse_target, action="append", default=[], help='Segmentation targets: "name" or "name,user_id,session_id".')
    input_group.add_argument("--run-ids", type=parsers.parse_list, default=None, help="List of run IDs.")

    # Parameters
    parameters_group = parser.add_argument_group("Parameters")
    parameters_group.add_argument("--tomo-alg", type=str, default="wbp", help="Tomogram reconstruction algorithm.")    
    parameters_group.add_argument("--radius-scale", type=float, default=0.7, help="Scale factor for object radius.")
    parameters_group.add_argument("--voxel-size", type=float, default=10, help="Voxel size for tomogram reconstruction.")    
    
    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--target-segmentation-name", type=str, default='targets', help="Name for the target segmentation.")
    output_group.add_argument("--target-user-id", type=str, default="octopi", help="User ID associated with the target segmentation.")
    output_group.add_argument("--target-session-id", type=str, default="1", help="Session ID for the target segmentation.")

    return parser.parse_args()

def cli():
    args = parse_args()

    # Save JSON with Parameters
    output_yaml = f'create-targets_{args.target_user_id}_{args.target_session_id}_{args.target_segmentation_name}.yaml'
    save_parameters(args, output_yaml)      

    if args.target:
        # If at least one --target is provided, call create_sub_train_targets
        create_sub_train_targets(
            config=args.config,
            pick_targets=args.target,
            seg_targets=args.seg_target,
            voxel_size=args.voxel_size,
            radius_scale=args.radius_scale,
            tomogram_algorithm=args.tomo_alg,
            target_segmentation_name=args.target_segmentation_name,
            target_user_id=args.target_user_id,
            target_session_id=args.target_session_id,
            run_ids=args.run_ids,
        )
    else:
        # If no --target is provided, call create_all_train_targets
        create_all_train_targets(
            config=args.config,
            seg_targets=args.seg_target,
            picks_session_id=args.picks_session_id,
            picks_user_id=args.picks_user_id,
            voxel_size=args.voxel_size,
            radius_scale=args.radius_scale,
            tomogram_algorithm=args.tomo_alg,
            target_segmentation_name=args.target_segmentation_name,
            target_user_id=args.target_user_id,
            target_session_id=args.target_session_id,
            run_ids=args.run_ids,
        )

def save_parameters(args, output_path: str):
    """
    Save parameters to a YAML file with subgroups for input, output, and parameters.
    Append to the file if it already exists.

    Args:
        args: Parsed arguments from argparse.
        output_path: Path to save the YAML file.
    """

    print('\nGenerating Target Segmentation Masks from the Following Copick-Query:')
    if args.picks_session_id is None or args.picks_user_id is None:
        print(f'    - {args.target}\n')
        input_group = {
            "config": args.config,
            "target": args.target,
        }
    else:
        print(f'    - {args.picks_session_id}, {args.picks_user_id}\n')
        input_group = {
            "config": args.config,
            "picks_session_id": args.picks_session_id,
            "picks_user_id": args.picks_user_id
        }
    if len(args.seg_target) > 0:
        input_group["seg_target"] = args.seg_target
        
    # Organize parameters into subgroups
    input_key = f'{args.target_user_id}_{args.target_session_id}_{args.target_segmentation_name}'
    new_entry = {
        input_key : {
            "input": input_group ,
            "parameters": {
                "radius_scale": args.radius_scale,
                "tomogram_algorithm": args.tomo_alg,
                "voxel_size": args.voxel_size,            
            }
        }
    }

     # Check if the YAML file already exists
    if os.path.exists(output_path):
        # Load the existing content
        with open(output_path, 'r') as f:
            try:
                existing_data = yaml.safe_load(f)
                if existing_data is None:
                    existing_data = {}  # Ensure it's a dictionary
                elif not isinstance(existing_data, dict):
                    raise ValueError("Existing YAML data is not a dictionary. Cannot update.")
            except yaml.YAMLError:
                existing_data = {}  # Treat as empty if the file is malformed
    else:
        existing_data = {}  # Initialize as empty list if the file does not exist

    # Append the new entry
    existing_data[input_key] = new_entry[input_key]

    # Save back to the YAML file
    io.save_parameters_yaml(existing_data, output_path)

if __name__ == "__main__":
    cli()
