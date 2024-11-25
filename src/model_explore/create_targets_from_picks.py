from copick_utils.segmentation.segmentation_from_picks import from_picks
from model_explore.pytorch import utils, io 
import copick_utils.writers.write as write
from collections import defaultdict
from typing import List, Tuple, Union
import argparse, copick
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

    generate_targets(root, train_targets, voxel_size, tomogram_algorithm, radius_scale,
                     target_segmentation_name, target_user_id, 
                     target_session_id, run_ids)


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

    generate_targets(root, target_objects, voxel_size, tomogram_algorithm, 
                     radius_scale, target_segmentation_name, target_user_id, 
                     target_session_id, run_ids )

def generate_targets(
    root,
    train_targets: dict,
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    radius_scale: float = 0.8,    
    target_segmentation_name: str = 'targets',
    target_user_name: str = 'monai',
    target_session_id: str = '1',
    run_ids: List[str] = None,
    ):
    """
    Generate segmentation targets from picks in CoPick configuration.

    Args:
        copick_config_path (str): Path to CoPick configuration file.
        picks_user_id (str): User ID associated with picks.
        picks_session_id (str): Session ID associated with picks.
        target_segmentation_name (str): Name for the target segmentation.
        target_user_name (str): User name associated with target segmentation.
        target_session_id (str): Session ID for the target segmentation.
        voxel_size (float): Voxel size for tomogram reconstruction.
        tomo_algorithm (str): Tomogram reconstruction algorithm.
        radius_scale (float): Scale factor for target object radius.
    """

    print('Creating Targets for the following objects:', ', '.join(train_targets.keys()))

    # Get Target Names
    target_names = list(train_targets.keys())

    # If runIDs are not provided, load all runs
    if run_ids is None:
        run_ids = [run.name for run in root.runs]

    # Iterate Over All Runs
    for runID in tqdm(run_ids):

        # Get Run
        numPicks = 0
        run = root.get_run(runID)

        # Get Tomogram 
        tomo = io.get_tomogram_array(run, voxel_size, tomo_algorithm)
        
        # Initialize Target Volume
        target = np.zeros(tomo.shape, dtype=np.uint8)

        # Generate Targets
        # Applicable segmentations
        query_seg = []
        for target_name in target_names:
            if not train_targets[target_name]["is_particle_target"]:            
                query_seg += run.get_segmentations(
                    name=target_name,
                    user_id=train_targets[target_name]["user_id"],
                    session_id=train_targets[target_name]["session_id"],
                    voxel_size=voxel_size
                )     

        # Add Segmentations to Target
        for seg in query_seg:
            classLabel = root.get_object(seg.name).label
            segvol = seg.numpy()
            # Set all non-zero values to the class label
            segvol[segvol > 0] = classLabel
            target[:] = segvol 

        # Applicable picks
        query = []
        for target_name in target_names:
            if train_targets[target_name]["is_particle_target"]:
                query += run.get_picks(
                    object_name=target_name,
                    user_id=train_targets[target_name]["user_id"],
                    session_id=train_targets[target_name]["session_id"],
                )

        for pick in query:
            numPicks += len(pick.points)
            target = from_picks(pick, 
                                target, 
                                train_targets[pick.pickable_object_name]['radius'] * radius_scale,
                                train_targets[pick.pickable_object_name]['label'],
                                voxel_size
                                )

        # Write Segmentation
        tqdm.write(f'Annotating {numPicks} picks in {runID}...')    
        write.segmentation(run, target, target_user_name, 
                           name = target_segmentation_name, session_id= target_session_id, 
                           voxel_size = voxel_size)

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
    parser = argparse.ArgumentParser(description="Generate segmentation targets from CoPick configurations.")

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

def main():
    args = parse_args() 

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

if __name__ == "__main__":
    main()