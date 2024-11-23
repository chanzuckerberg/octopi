from copick_utils.segmentation.segmentation_from_picks import from_picks
from model_explore.pytorch import utils, io 
import copick_utils.writers.write as write
from collections import defaultdict
import argparse, copick
from tqdm import tqdm
import numpy as np

def create_sub_train_targets(
    config: str,
    train_targets: dict,
    seg_targets: dict,
    run_ids: List[str],
    voxel_size: float,
    tomogram_algorithm: str,
    target_segmentation_name: str,
    target_user_id: str,
    target_session_id: str,
    ):

    root = copick.from_file(config)

    train_targets = {}
    for t in target:
        obj_name, user_id, session_id, radius = t
        info = {
            "label": root.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "is_particle_target": True,
        }
        train_targets[obj_name] = info

    for s in seg_target:
        obj_name, user_id, session_id = s
        info = {
            "label": root.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": None,       
            "is_particle_target": False,                 
        }
        train_targets[obj_name] = info

    import pdb; pdb.set_trace()

    generate_targets(root, train_targets, 
                     target_segmentation_name, target_user_id, 
                     target_session_id, voxel_size, tomogram_algorithm)



def create_all_train_targets(
    root,
    seg_targets: dict,
    run_ids: List[str],
    voxel_size: float,
    tomogram_algorithm: str,
    target_segmentation_name: str,
    target_user_id: str,
    target_session_id: str,
    ):

    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius

    # Todo: Create dictionary for segmentation targets
    for s in seg_target:
        obj_name, user_id, session_id = s
        info = {
            "label": root.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": None,       
            "is_particle_target": False,                 
        }
        train_targets[obj_name] = info

    import pdb; pdb.set_trace()

    generate_targets(root, train_targets, 
                     target_segmentation_name, target_user_id, 
                     target_session_id, voxel_size, tomogram_algorithm)

def generate_targets(
    root,
    train_targets: dict,
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    radius_scale: float = 0.8,    
    target_segmentation_name: str = 'targets',
    target_user_name: str = 'monai',
    target_session_id: str = '1',
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

    # Get Target Names
    target_names = list(train_targets.keys())

    # Iterate Over All Runs
    for run in tqdm(root.runs):

        # Get Tomogram 
        tomo = io.get_tomogram_array(run, voxel_size, tomo_algorithm)
        
        # Initialize Target Volume
        target = np.zeros(tomo.shape, dtype=np.uint8)

        # Generate Targets
        for pickable_object in root.pickable_objects:

            # Applicable segmentations
            query_seg = []
            for target_name in target_names:
                if not train_targets[target_name]["is_particle_target"]:            
                    query_seg += run.get_segmentations(
                        name=target_name,
                        user_id=train_targets[target_name]["user_id"],
                        session_id=train_targets[target_name]["session_id"],
                        voxel_size=voxel_size,
                        is_multilabel=False,
                    )     

             # Add Segmentations to Target
            for seg in query_seg:
                classLabel = root.get_object(seg.name).label
                segvol = seg.numpy()
                # Set all non-zero values to the class label
                segvol[segvol > 0] = classLabel
                target_vol[:] = segvol 

            # Read Picks for Associated Run and Object
            pick = run.get_picks(object_name=pickable_object.name, user_id=picks_user_id, session_id=picks_session_id)
            
            # Todo: Check if There is More than One Pick Available
            if len(pick):  
                target = from_picks(pick[0], 
                                    target, 
                                    target_objects[pickable_object.name]['radius'] * radius_scale,
                                    target_objects[pickable_object.name]['label'],
                                    voxel_size
                                    )
                
        # Write Segmentation
        write.segmentation(run, target, target_user_name, 
                           name = target_segmentation_name, session_id= target_session_id, 
                           voxel_size = voxel_size)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation targets from CoPick configurations.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands for creating targets.")

    # Subcommand: create_all
    create_all_parser = subparsers.add_parser("create_all", help="Create targets for all objects.")
    create_all_parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    create_all_parser.add_argument("--run-ids", type=str, nargs="+", default=None, help="List of run IDs.")
    create_all_parser.add_argument("--voxel-size", type=float, default=10, help="Voxel size for tomogram reconstruction.")
    create_all_parser.add_argument("--tomogram-algorithm", type=str, default="wbp", help="Tomogram reconstruction algorithm.")
    create_all_parser.add_argument("--radius-scale", type=float, default=0.8, help="Scale factor for object radius.")
    create_all_parser.add_argument("--target-segmentation-name", type=str, required=True, help="Name for the target segmentation.")
    create_all_parser.add_argument("--target-user-id", type=str, default="monai", help="User ID associated with the target segmentation.")
    create_all_parser.add_argument("--target-session-id", type=str, default="1", help="Session ID for the target segmentation.")

    # Subcommand: create_sub
    create_sub_parser = subparsers.add_parser("create_sub", help="Create targets for specific objects.")
    create_sub_parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    create_sub_parser.add_argument("--target", type=str, nargs="+", required=True, help='Target specifications: "name,user_id,session_id,radius".')
    create_sub_parser.add_argument("--seg-target", type=str, nargs="+", required=False, help='Segmentation targets: "name,user_id,session_id".')
    create_sub_parser.add_argument("--voxel-size", type=float, default=10, help="Voxel size for tomogram reconstruction.")
    create_sub_parser.add_argument("--tomogram-algorithm", type=str, default="wbp", help="Tomogram reconstruction algorithm.")
    create_sub_parser.add_argument("--target-segmentation-name", type=str, required=True, help="Name for the target segmentation.")
    create_sub_parser.add_argument("--target-user-id", type=str, default="monai", help="User ID associated with the target segmentation.")
    create_sub_parser.add_argument("--target-session-id", type=str, default="1", help="Session ID for the target segmentation.")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "create_all":
        create_all_train_targets(
            config=args.config,
            run_ids=args.run_ids,
            voxel_size=args.voxel_size,
            tomogram_algorithm=args.tomogram_algorithm,
            target_segmentation_name=args.target_segmentation_name,
            target_user_id=args.target_user_id,
            target_session_id=args.target_session_id,
        )
    elif args.command == "create_sub":
        create_sub_train_targets(
            config=args.config,
            target=args.target,
            seg_target=args.seg_target,
            voxel_size=args.voxel_size,
            tomogram_algorithm=args.tomogram_algorithm,
            target_segmentation_name=args.target_segmentation_name,
            target_user_id=args.target_user_id,
            target_session_id=args.target_session_id,
        )
    else:
        print("Invalid command. Use --help for available commands.")


if __name__ == "__main__":
    main()