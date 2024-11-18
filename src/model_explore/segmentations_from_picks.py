from copick_utils.segmentation import target_generator
import copick_utils.writers.write as write
from collections import defaultdict
import argparse, copick
from tqdm import tqdm
import numpy as np

def generate_targets_from_picks(
    copick_config_path: str,
    picks_user_id: str,
    picks_session_id: str,
    target_segmentation_name: str,
    target_user_name: str = 'monai',
    target_session_id: str = '1',
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    target_scale: float = 0.8
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
        target_scale (float): Scale factor for target object radius.
    """

    root = copick.from_file(copick_config_path)

    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius

    for run in tqdm(root.runs):
        tomo = run.get_voxel_spacing(voxel_size)
        tomo = tomo.get_tomogram(tomo_algorithm).numpy()
        target = np.zeros(tomo.shape, dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name=pickable_object.name, user_id=picks_user_id, session_id=picks_session_id)
            if len(pick):  
                target = target_generator.from_picks(pick[0], 
                                                    target, 
                                                    target_objects[pickable_object.name]['radius'] * target_scale,
                                                    target_objects[pickable_object.name]['label']
                                                    )
                
        # Write Segmentation
        write.segmentation(run, target, target_user_name, 
                           name = target_segmentation_name, session_id= target_session_id, 
                           voxel_size = voxel_size)

def cli():

    parser = argparse.ArgumentParser(
        description = "Generate picks segmenations from copick files."
    )

    parser = argparse.ArgumentParser(description="Generate picks segmentations from CoPick files.")

    # Required arguments
    parser.add_argument("--copick_config_path", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--picks_user_id", type=str, required=True, help="User ID associated with picks.")
    parser.add_argument("--target_segmentation_name", type=str, required=True, help="Name for the target segmentation.")

    # Optional arguments
    parser.add_argument("--picks_session_id", type=str, required=True, help="Session ID associated with picks.")
    parser.add_argument("--target_user_name", type=str, default="monai", help="User name associated with target segmentation.")
    parser.add_argument("--target_session_id", type=str, default="1", help="Session ID for the target segmentation.")
    parser.add_argument("--voxel_size", type=float, default=10.0, help="Voxel size for tomogram reconstruction.")
    parser.add_argument("--tomo_algorithm", type=str, default="wbp", help="Tomogram reconstruction algorithm.")
    parser.add_argument("--target_scale", type=float, default=0.8, help="Scale factor for target object radius.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_targets_from_picks(
        copick_config_path=args.copick_config_path,
        picks_user_id=args.picks_user_id,
        picks_session_id=args.picks_session_id,
        target_segmentation_name=args.target_segmentation_name,
        target_user_name=args.target_user_name,
        target_session_id=args.target_session_id,
        voxel_size=args.voxel_size,
        tomo_algorithm=args.tomo_algorithm,
        target_scale=args.target_scale,
    )

# if __name__ == '__main__':
#     args = get_args()
#     root = copick.from_file(args.copick_config_path)

#     voxel_spacing = 10
#     target_objects = defaultdict(dict)
#     for object in root.pickable_objects:
#         if object.is_particle:
#             target_objects[object.name]['label'] = object.label
#             target_objects[object.name]['radius'] = object.radius

#     for run in tqdm(root.runs):
#         tomo = run.get_voxel_spacing(10)
#         tomo = tomo.get_tomogram('wbp').numpy()
#         target = np.zeros(tomo.shape, dtype=np.uint8)
#         for pickable_object in root.pickable_objects:
#             pick = run.get_picks(object_name=pickable_object.name, user_id="data-portal")
#             if len(pick):  
#                 target = target_generator.from_picks(pick[0], 
#                                                     target, 
#                                                     target_objects[pickable_object.name]['radius'] * 0.8,
#                                                     target_objects[pickable_object.name]['label']
#                                                     )
#         write.segmentation(run, target, args.user_name, segmentationName=args.segmentation_name)