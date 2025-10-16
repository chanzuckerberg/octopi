from octopi.processing.segmentation_from_picks import from_picks
from copick_utils.io import readers, writers
import zarr, os, yaml, copick, io
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

        # Get Target Shape
        vs = run.get_voxel_spacing(voxel_size)
        if vs is None:
            print(f"Warning: skipping run {runID} with no voxel spacing {voxel_size}")
            continue
        tomo = vs.get_tomogram(tomo_algorithm)
        if tomo is None:
            print(f"Warning: skipping run {runID} with no tomogram {tomo_algorithm}")
            continue
        
        # Initialize Target Volume
        loc = tomo.zarr()
        shape = zarr.open(loc)['0'].shape
        target = np.zeros(shape, dtype=np.uint8)

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
            target = np.maximum(target, segvol)

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
        if target.max() > 0:
            tqdm.write(f'Annotating {numPicks} picks in {runID}...')    
            writers.segmentation(run, target, target_user_name, 
                               name = target_segmentation_name, session_id= target_session_id, 
                               voxel_size = voxel_size)
    
    # Save Parameters
    args = {
        "config": root.config.path,
        "picks_session_id": target_session_id,
        "picks_user_id": target_user_name,
        "seg_target": target_segmentation_name,
        "radius_scale": radius_scale,
        "tomo_algorithm": tomo_algorithm,
        "voxel_size": voxel_size,
    }
    output_path = os.path.join(root.config.overlay_root, 'logs', f'create-targets_{target_user_name}_{target_session_id}_{target_segmentation_name}.yaml')
    save_parameters(args, output_path)
    print('Creation of targets complete!')



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
    root = copick.from_file(args.config)
    basepath = os.path.join(root.config.overlay_root, 'logs')
    os.makedirs(basepath, exist_ok=True)
    output_path = os.path.join(
        basepath, 
        f'create-targets_{args.target_user_id}_{args.target_session_id}_{args.target_segmentation_name}.yaml')
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
