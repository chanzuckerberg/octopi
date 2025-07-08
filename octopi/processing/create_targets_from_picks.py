from octopi.processing.segmentation_from_picks import from_picks
from copick_utils.io import readers, writers
from typing import List
from tqdm import tqdm
import numpy as np

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

    # Default session ID to 1 if not provided
    if target_session_id is None:
        target_session_id = '1'

    print('Creating Targets for the following objects:', ', '.join(train_targets.keys()))

    # Get Target Names
    target_names = list(train_targets.keys())

    # If runIDs are not provided, load all runs
    if run_ids is None:
        run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is not None]
        skipped_run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is None]
        
        if skipped_run_ids:
            print(f"Warning: skipping runs with no voxel spacing {voxel_size}: {skipped_run_ids}")

    # Iterate Over All Runs
    for runID in tqdm(run_ids):

        # Get Run
        numPicks = 0
        run = root.get_run(runID)

        # Get Tomogram 
        tomo = readers.tomogram(run, voxel_size, tomo_algorithm)
        
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

        # Filter out empty picks
        query = [pick for pick in query if pick.points is not None]

        # Add Picks to Target  
        for pick in query:
            numPicks += len(pick.points)
            target = from_picks(pick, 
                                target, 
                                train_targets[pick.pickable_object_name]['radius'] * radius_scale,
                                train_targets[pick.pickable_object_name]['label'],
                                voxel_size
                                )

        # Write Segmentation for non-empty targets
        if target.max() > 0 and numPicks > 0:
            tqdm.write(f'Annotating {numPicks} picks in {runID}...')    
            writers.segmentation(run, target, target_user_name, 
                               name = target_segmentation_name, session_id= target_session_id, 
                               voxel_size = voxel_size)
    print('Creation of targets complete!')
