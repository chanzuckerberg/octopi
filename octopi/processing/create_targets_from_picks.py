from octopi.processing.segmentation_from_picks import from_picks
from copick_utils.io import readers, writers
import zarr, os, yaml, copick
from octopi.utils import io
from typing import List
from tqdm import tqdm
import numpy as np

def print_target_summary(train_targets: dict, target_segmentation_name: str, maxval: int):
    """
    Print a summary of the target volume structure.
    """
    print("\n" + "="*60)
    print("TARGET VOLUME SUMMARY")
    print("="*60)
    print(f"Segmentation name: {target_segmentation_name}")
    print(f"Total classes: {len(train_targets) + 1} (including background)")
    print("\nLabel Index → Object Name (Type):")
    print(f"  {0:3d} → background")
    
    # Sort by label for display
    sorted_targets = sorted(train_targets.items(), key=lambda x: x[1]['label'])
    for name, info in sorted_targets:
        obj_type = "particle" if info['is_particle_target'] else "segmentation"
        radius_info = f", radius={info['radius']:.1f}Å" if info['radius'] else ""
        print(f"  {info['label']:3d} → {name} ({obj_type}{radius_info})")
    
    print("="*60)
    print(f"💡 Use --num-classes {maxval + 1} when training with this target")
    print("="*60 + "\n")

def generate_targets(
    config,
    train_targets: dict,
    voxel_size: float = 10,
    tomo_algorithm: str = 'wbp',
    radius_scale: float = 0.8,    
    target_segmentation_name: str = 'targets',
    target_user_name: str = 'octopi',
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
    root = copick.from_file(config)
    if target_session_id is None:
        target_session_id = '1'

    # Print target summary
    print('🔄 Creating Targets for the following objects:', ', '.join(train_targets.keys()))

    # Get Target Names
    target_names = list(train_targets.keys())

    # If runIDs are not provided, load all runs
    if run_ids is None:
        run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is not None]
        skipped_run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is None]
        
        if skipped_run_ids:
            print(f"⚠️ Warning: skipping runs with no voxel spacing {voxel_size}: {skipped_run_ids}")

    # Iterate Over All Runs
    maxval = -1
    for runID in tqdm(run_ids):

        # Get Run
        numPicks = 0
        run = root.get_run(runID)

        # Get Target Shape
        vs = run.get_voxel_spacing(voxel_size)
        if vs is None:
            print(f"⚠️ Warning: skipping run {runID} with no voxel spacing {voxel_size}")
            continue
        tomo = vs.get_tomogram(tomo_algorithm)
        if tomo is None:
            print(f"⚠️ Warning: skipping run {runID} with no tomogram {tomo_algorithm}")
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
            classLabel = train_targets[seg.name]['label']
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
            tqdm.write(f'📝 Annotating {numPicks} picks in {runID}...')    
            writers.segmentation(run, target, target_user_name, 
                               name = target_segmentation_name, session_id= target_session_id, 
                               voxel_size = voxel_size)
        if target.max() > maxval:
            maxval = target.max()
    
    print('✅ Creation of targets complete!')

    # Save Parameters
    overlay_root = io.remove_prefix(root.config.overlay_root)
    basepath = os.path.join(overlay_root, 'logs')
    os.makedirs(basepath, exist_ok=True)
    labels = {name: info['label'] for name, info in train_targets.items()}
    args = {
        "config": config,
        "train_targets": train_targets,
        "radius_scale": radius_scale,
        "tomo_algorithm": tomo_algorithm,
        "target_name": target_segmentation_name,
        "target_user_name": target_user_name,
        "target_session_id": target_session_id,
        "voxel_size": voxel_size,
        "labels": labels,
    }
    target_query = f'{target_user_name}_{target_session_id}_{target_segmentation_name}'
    print(f'💾 Saving parameters to {basepath}/targets-{target_query}.yaml')
    save_parameters(args, basepath, target_query)

    # Print Target Summary
    print_target_summary(train_targets, target_segmentation_name, maxval)

def save_parameters(args, basepath: str, target_query: str):
    """
    Save parameters to a YAML file with subgroups for input, output, and parameters.
    Append to the file if it already exists.

    Args:
        args: Parsed arguments from argparse.
        basepath: Path to save the YAML file.
        target_query: Query string for target identification.
    """
    # Prepare input group
    keys = ['user_id', 'session_id']
    input_group = {
        "config": args['config'],
        "labels": {name: info['label'] for name, info in args['train_targets'].items()},  # <-- Added comma here
        "targets": {name: {k: info[k] for k in keys} for name, info in args['train_targets'].items()}
    }
        
    # Organize parameters into subgroups
    new_entry = {
        "input": input_group,
        "parameters": {
            "radius_scale": args["radius_scale"],
            "tomogram_algorithm": args["tomo_algorithm"],
            "voxel_size": args["voxel_size"],
        }
    }

    # Check if the YAML file already exists
    output_path = os.path.join(
        basepath, 
        f'targets-{args["target_user_name"]}_{args["target_session_id"]}_{args["target_name"]}.yaml')
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
        existing_data = {}  # Initialize as empty dictionary if the file does not exist

    # Save back to the YAML file
    io.save_parameters_yaml(new_entry, output_path)