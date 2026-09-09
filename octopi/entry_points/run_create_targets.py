from typing import List, Tuple, Union
from collections import defaultdict
from octopi.utils import parsers
import rich_click as click

def create_sub_train_targets(
    config: str,
    targets: List[Tuple[str, Union[str, None], Union[str, None]]],
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

    # Create dictionary for all targets, preserving the order the objects were
    # given so that sequential labels reflect the true intended order even when
    # particle and segmentation targets are mixed together.
    train_targets = defaultdict(dict)

    label = 1
    for t in targets:
        # Parse the target
        obj_name, user_id, session_id = t
        obj = root.get_object(obj_name)

        # Check if the object is valid
        if obj is None:
            print(f'Warning - Skipping Target: "{obj_name}", as it is not a valid name in the config file.')
            continue

        if obj_name in train_targets:
            print(f'Warning - Skipping Target: "{obj_name}, {user_id}, {session_id}", as it has already been added to the target list.')
            continue

        # Determine whether this is a particle (picks) or continuous segmentation
        # target directly from the CoPick config, rather than which CLI flag was used.
        info = {
            "label": label,
            "user_id": user_id,
            "session_id": session_id,
            "is_particle_target": obj.is_particle,
            "radius": obj.radius if obj.is_particle else None,
        }
        train_targets[obj_name] = info
        label += 1

    create_targets.generate_targets(
        config, train_targets, voxel_size, tomogram_algorithm, radius_scale,
        target_segmentation_name, target_user_id,
        target_session_id, run_ids
    )


def create_all_train_targets(
    config: str,
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

    # Create dictionary for all pickable objects, auto-detecting whether each
    # is a particle (picks) or continuous segmentation target from the config.
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        info = {
            "label": object.label,
            "radius": object.radius,
            "user_id": picks_user_id,
            "session_id": picks_session_id,
            "is_particle_target": object.is_particle,
        }
        target_objects[object.name] = info

    create_targets.generate_targets(
        config, target_objects, voxel_size, tomogram_algorithm,
        radius_scale, target_segmentation_name, target_user_id,
        target_session_id, run_ids
    )


@click.command('create-targets', no_args_is_help=True)
# Output Arguments
@click.option('-turi', '--target-uri', type=str, default="targets:octopi/1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target query as 'name', 'name:user_id', or 'name:user_id/session_id'. Default 'targets:octopi/1'.")
# Parameters
@click.option('-uri', '--tomo-uri', type=str, default="wbp@10.0",
              help="Tomogram URI for target dimensions (tomo-alg@voxel-size)")
@click.option('-rs', '--radius-scale', type=float, default=0.7,
              help="Scale factor for object radius")
# Input Arguments
@click.option('--run-ids', '-runs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of run IDs")
@click.option('--picks-user-id', '-puid', type=str, default=None,
              help="User ID associated with the picks")
@click.option('--picks-session-id', '-psid', type=str, default=None,
              help="Session ID for the picks")
@click.option('-t', '--target', type=str, multiple=True,
              callback=lambda ctx, param, value: [parsers.parse_target(v) for v in value] if value else None,
              help='Target object(s) — particle picks or continuous segmentations, auto-detected from the '
                   'CoPick config: "name", "name:user_id/session_id", or the legacy "name,user_id,session_id". '
                   'Labels are assigned sequentially in the order the flags are given, so a mix of particle '
                   'and segmentation targets keeps its true order.')
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the CoPick configuration file")
def cli(config, target, picks_session_id, picks_user_id, run_ids,
        tomo_uri, radius_scale, target_uri):
    """
    Generate segmentation targets from CoPick configurations.

    This tool allows users to specify target labels for training in two ways:

    1. Manual Specification: Define a subset of pickable objects using --target name or --target name:user_id/session_id.
       Each --target can be a particle pick set or a continuous segmentation (e.g. membrane) — the type is
       auto-detected from the CoPick config, so no separate flag is needed.

    2. Automated Query: Provide --picks-session-id and/or --picks-user-id to automatically retrieve all pickable objects

    Example Usage:

        Manual: octopi create-targets --config config.json --target ribosome --target membrane:membrane-seg/1 --tomo-uri wbp@10.0

        Automated: octopi create-targets --config config.json --picks-session-id 123 --picks-user-id 456 --tomo-uri wbp@10.0
    """

    # Parse the Tomogram URI
    if '@' not in tomo_uri:
        raise ValueError("Tomogram URI must contain '@' for voxel size, e.g. 'wbp@10.0'.")
    tomo_alg, voxel_size = tomo_uri.split('@')
    voxel_size = float(voxel_size)

    # Parse the Target URI
    target_segmentation_name, target_user_id, target_session_id = target_uri

    # Print Summary To User
    print('\n⚙️ Generating Target Segmentation Masks from the Following Copick-Query:')
    if target is not None and len(target) > 0:
        print(f'    - Targets: {target}')
    elif picks_user_id is not None or picks_session_id is not None:
        print(f'    -  UserID: {picks_user_id} -- SessionID: {picks_session_id}')
    print()

    # Check if any --target was provided
    if target is not None and len(target) > 0:
        # If at least one --target is provided, call create_sub_train_targets
        create_sub_train_targets(
            config=config,
            targets=target,
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
