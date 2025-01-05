from model_explore import create_targets_from_picks as create_targets
from model_explore import utils, io 
import copick_utils.writers.write as write
from collections import defaultdict
from typing import List, Tuple, Union
import argparse, copick, json, os
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
        obj_name, user_id, session_id = t 
        info = {
            "label": root.get_object(obj_name).label,
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
            "user_id": picks_session_id,
            "session_id": picks_user_id,
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
    Parse command line arguments for target generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate segmentation targets from CoPick configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--target", type=utils.parse_target, action="append", default=None, help='Target specifications: "name,user_id,session_id".')
    parser.add_argument("--seg-target", type=utils.parse_target, action="append", default=[], help='Segmentation targets: "name" or "name,user_id,session_id".')
    parser.add_argument("--voxel-size", type=float, default=10, help="Voxel size for tomogram reconstruction.")
    parser.add_argument("--radius-scale", type=float, default=0.8, help="Scale factor for object radius.")
    parser.add_argument("--tomogram-algorithm", type=str, default="wbp", help="Tomogram reconstruction algorithm.")
    parser.add_argument("--picks-session-id", type=str, default=None, help="Session ID for the picks.")
    parser.add_argument("--picks-user-id", type=str, default=None, help="User ID associated with the picks.")
    parser.add_argument("--target-segmentation-name", type=str, default=None, help="Name for the target segmentation.")
    parser.add_argument("--target-user-id", type=str, default="monai", help="User ID associated with the target segmentation.")
    parser.add_argument("--target-session-id", type=str, default="1", help="Session ID for the target segmentation.")
    parser.add_argument("--run-ids", type=utils.parse_list, default=None, help="List of run IDs.")

    return parser.parse_args()

def cli():
    args = parse_args()

    # Save JSON with Parameters
    output_json = f'create-targets_{args.target_user_id}_{args.target_session_id}_{args.target_segmentation_name}.json'
    save_parameters_json(args, output_json)      

    if args.target:
        # If at least one --target is provided, call create_sub_train_targets
        create_sub_train_targets(
            config=args.config,
            pick_targets=args.target,
            seg_targets=args.seg_target,
            voxel_size=args.voxel_size,
            radius_scale=args.radius_scale,
            tomogram_algorithm=args.tomogram_algorithm,
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
            tomogram_algorithm=args.tomogram_algorithm,
            target_segmentation_name=args.target_segmentation_name,
            target_user_id=args.target_user_id,
            target_session_id=args.target_session_id,
            run_ids=args.run_ids,
        )

def save_parameters_json(args, output_path: str):
    """
    Save parameters to a JSON file with subgroups for input, output, and parameters.
    Append to the file if it already exists.

    Args:
        args: Parsed arguments from argparse.
        output_path: Path to save the JSON file.
    """
    # Organize parameters into subgroups
    new_entry = {
        "input": {
            "config": args.config,
            "target": args.target,
            "seg_target": args.seg_target,
            "picks_session_id": args.picks_session_id,
            "picks_user_id": args.picks_user_id
        },
        "output": {
            "target_segmentation_name": args.target_segmentation_name,
            "target_user_id": args.target_user_id,
            "target_session_id": args.target_session_id,
        },
        "parameters": {
            "radius_scale": args.radius_scale,
            "tomogram_algorithm": args.tomogram_algorithm,
            "voxel_size": args.voxel_size,            
        }
    }

    # Check if the JSON file already exists
    if os.path.exists(output_path):
        # Load the existing content
        with open(output_path, 'r') as f:
            try:
                existing_data = json.load(f)
                # Ensure it's a list to append to
                if not isinstance(existing_data, list):
                    raise ValueError("Existing JSON data is not a list. Cannot append.")
            except json.JSONDecodeError:
                existing_data = []  # Treat as empty if the file is malformed
    else:
        existing_data = []  # No file, start with an empty list

    # Append the new entry
    existing_data.append(new_entry)

    # Save back to the JSON file
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

if __name__ == "__main__":
    main()
