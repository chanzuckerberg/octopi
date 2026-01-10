from typing import List, Tuple, Union
from collections import defaultdict
from octopi.utils import parsers
import rich_click as click

def create_sub_train_targets(
    config: str,
    pick_targets: List[Tuple[str, Union[str, None], Union[str, None]]],
    seg_targets: List[Tuple[str, Union[str, None], Union[str, None]]],
    voxel_size: float,
    radius_scale: float,    
    tomogram_algorithm: str,
    target_segmentation_name: str,
    target_user_id: str,
    target_session_id: str,
    run_ids: List[str],    
    ):
    import octopi.processing.create_targets_from_picks as create_targets
    import copick

    # Load Copick Project 
    root = copick.from_file(config)

    # Create empty dictionary for all targets
    train_targets = defaultdict(dict)

    # Create dictionary for particle targets
    value = 1 
    for t in pick_targets:
        # Parse the target
        obj_name, user_id, session_id = t 
        obj = root.get_object(obj_name)

        # Check if the object is valid
        if obj is None:
            print(f'Warning - Skipping Particle Target: "{obj_name}", as it is not a valid name in the config file.')
            continue

        if obj_name in train_targets:
            print(f'Warning - Skipping Particle Target: "{obj_name}, {user_id}, {session_id}", as it has already been added to the target list.')
            continue
        
        # Get the label and radius of the object
        label = value # Assign labels sequentially
        info = {
            "label": label,
            "user_id": user_id,
            "session_id": session_id,
            "is_particle_target": True,
            "radius": root.get_object(obj_name).radius,
        }
        train_targets[obj_name] = info
        value += 1

    # Create dictionary for segmentation targets
    train_targets = add_segmentation_targets(root, seg_targets, train_targets, value)   

    create_targets.generate_targets(
        config, train_targets, voxel_size, tomogram_algorithm, radius_scale,
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
    import octopi.processing.create_targets_from_picks as create_targets
    import copick

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
        config, target_objects, voxel_size, tomogram_algorithm, 
        radius_scale, target_segmentation_name, target_user_id, 
        target_session_id, run_ids 
    )

def add_segmentation_targets(
    root,
    seg_targets,
    train_targets: dict,
    start_value: int = -1
    ):

    # Create dictionary for segmentation targets
    for s in seg_targets:

        # Parse Segmentation Target
        obj_name, user_id, session_id = s

        # Add Segmentation Target
        if start_value > 0:
            value = start_value
            start_value += 1
        else:
            value = root.get_object(obj_name).label

        try:
            info = {
                "label": value,
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


@click.command('create-targets')
# Output Arguments
@click.option('-sid', '--target-session-id', type=str, default="1",
              help="Session ID for the target segmentation")
@click.option('-uid','--target-user-id', type=str, default="octopi",
              help="User ID associated with the target segmentation")
@click.option('-name', '--target-segmentation-name', type=str, default='targets',
              help="Name for the target segmentation")
# Parameters
@click.option('-vs', '--voxel-size', type=float, default=10,
              help="Voxel size for tomogram reconstruction")
@click.option('-rs', '--radius-scale', type=float, default=0.7,
              help="Scale factor for object radius")
@click.option('-alg', '--tomo-alg', type=str, default="wbp",
              help="Tomogram reconstruction algorithm")
# Input Arguments
@click.option('--run-ids', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of run IDs")
@click.option('--seg-target', type=str, multiple=True,
              callback=lambda ctx, param, value: [parsers.parse_target(v) for v in value] if value else [],
              help='Segmentation targets: "name" or "name,user_id,session_id"')
@click.option('--picks-user-id', type=str, default=None,
              help="User ID associated with the picks")
@click.option('--picks-session-id', type=str, default=None,
              help="Session ID for the picks")
@click.option('-t', '--target', type=str, multiple=True,
              callback=lambda ctx, param, value: [parsers.parse_target(v) for v in value] if value else None,
              help='Target specifications: "name" or "name,user_id,session_id"')
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the CoPick configuration file")
def cli(config, target, picks_session_id, picks_user_id, seg_target, run_ids,
        tomo_alg, radius_scale, voxel_size,
        target_segmentation_name, target_user_id, target_session_id):
    """
    Generate segmentation targets from CoPick configurations.

    This tool allows users to specify target labels for training in two ways:

    1. Manual Specification: Define a subset of pickable objects using --target name or --target name,user_id,session_id

    2. Automated Query: Provide --picks-session-id and/or --picks-user-id to automatically retrieve all pickable objects

    Example Usage:

        Manual: octopi create-targets --config config.json --target ribosome --target apoferritin,123,456

        Automated: octopi create-targets --config config.json --picks-session-id 123 --picks-user-id 456
    """

    # Print Summary To User
    print('\n⚙️ Generating Target Segmentation Masks from the Following Copick-Query:')
    if target is not None and len(target) > 0:
        print(f'    - Targets: {target}\n')
    else:
        print(f'    -  UserID: {picks_user_id} -- SessionID: {picks_session_id} \n')

    # Check if either target or seg_target is provided
    if (target is not None and len(target) > 0) or seg_target:
        # If at least one --target is provided, call create_sub_train_targets
        create_sub_train_targets(
            config=config,
            pick_targets=target if target else [],
            seg_targets=seg_target,
            voxel_size=voxel_size,
            radius_scale=radius_scale,
            tomogram_algorithm=tomo_alg,
            target_segmentation_name=target_segmentation_name,
            target_user_id=target_user_id,
            target_session_id=target_session_id,
            run_ids=run_ids,
        )
    else:
        # If no --target is provided, call create_all_train_targets
        create_all_train_targets(
            config=config,
            seg_targets=seg_target,
            picks_session_id=picks_session_id,
            picks_user_id=picks_user_id,
            voxel_size=voxel_size,
            radius_scale=radius_scale,
            tomogram_algorithm=tomo_alg,
            target_segmentation_name=target_segmentation_name,
            target_user_id=target_user_id,
            target_session_id=target_session_id,
            run_ids=run_ids,
        )


if __name__ == "__main__":
    cli()