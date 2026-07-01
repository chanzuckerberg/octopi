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
    label_space: str = 'model',
    from_model: str = None,
    ):
    import octopi.processing.create_targets_from_picks as create_targets
    import copick

    # Load Copick Project
    root = copick.from_file(config)

    # Create empty dictionary for all targets
    train_targets = defaultdict(dict)

    # Create dictionary for particle targets. Labels are assigned sequentially in
    # argument order (legacy model/dense space); `finalize_label_space` may re-express
    # them in copick global space below when `--label-space copick` is requested.
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

        # Assign labels sequentially
        label = value
        info = {
            "label": label,
            "user_id": user_id,
            "session_id": session_id,
            "is_particle_target": True,
            "radius": root.get_object(obj_name).radius,
        }
        train_targets[obj_name] = info
        value += 1

    # Create dictionary for segmentation targets (sequential, continuing the count)
    train_targets = add_segmentation_targets(root, seg_targets, train_targets, value)

    # Optionally re-express targets in copick GLOBAL label space (explicit label_space).
    label_space = finalize_label_space(root, train_targets, label_space, from_model)

    create_targets.generate_targets(
        config, train_targets, voxel_size, tomogram_algorithm, radius_scale,
        target_segmentation_name, target_user_id,
        target_session_id, run_ids, label_space=label_space,
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
    label_space: str = 'model',
    from_model: str = None,
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

    # Optionally re-express targets in copick GLOBAL label space (explicit label_space).
    label_space = finalize_label_space(root, target_objects, label_space, from_model)

    create_targets.generate_targets(
        config, target_objects, voxel_size, tomogram_algorithm,
        radius_scale, target_segmentation_name, target_user_id,
        target_session_id, run_ids, label_space=label_space,
    )

def add_segmentation_targets(
    root,
    seg_targets,
    train_targets: dict,
    start_value: int = -1,
    ):

    # Create dictionary for segmentation targets
    for s in seg_targets:

        # Parse Segmentation Target
        obj_name, user_id, session_id = s

        # Assign a sequential label (continuing the particle count) or fall back to the
        # object's global label; `finalize_label_space` re-expresses these when needed.
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


def finalize_label_space(root, train_targets, label_space, from_model):
    """
    Optionally re-express the target labels in copick GLOBAL label space.

    Default (``label_space='model'``): no-op — the recorded ``label`` stays the model/dense
    value and is what gets painted into the segmentation (legacy behavior, unchanged).

    ``label_space='copick'``: the segmentation is painted with each object's copick GLOBAL
    label (``root.get_object(name).label``), while the recorded ``label`` becomes the
    model's dense channel (``model_label``). Channels are minted by ascending global label,
    or, with ``from_model``, inherited per-name from a pretrained model config so fine-tuning
    keeps the pretrained channel<->object binding. Returns the marker to persist in the
    targets YAML (``'copick'`` or ``None``).
    """
    if label_space != 'copick':
        return None

    from octopi.utils import io

    names = list(train_targets.keys())
    copick_labels = {n: root.get_object(n).label for n in names}

    if from_model:
        pretrained = io.load_yaml(from_model).get('labels') or {}
        missing = [n for n in names if n not in pretrained]
        if missing:
            raise ValueError(
                f"--from-model {from_model} does not define model labels for {missing}; cannot "
                f"align channels for fine-tuning (fine-tuning onto a new object set is out of scope)." )
        model_labels = {n: int(pretrained[n]) for n in names}
    else:
        # Mint contiguous dense channels 1..K by ascending copick global label.
        order = sorted(names, key=lambda n: copick_labels[n])
        model_labels = {n: i + 1 for i, n in enumerate(order)}

    for n in names:
        train_targets[n]['paint_label'] = copick_labels[n]   # canonical copick label on disk
        train_targets[n]['label'] = model_labels[n]          # recorded model (dense) channel
    return 'copick'


@click.command('create-targets', no_args_is_help=True)
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
@click.option('--label-space', type=click.Choice(['model', 'copick']), default='model',
              help="Label space of the persisted targets. 'model' (default): legacy dense/sequential "
                   "labels. 'copick': paint canonical copick GLOBAL labels and record model channels "
                   "(name -> model_label) plus a `label_space: copick` marker in the targets YAML.")
@click.option('--from-model', type=click.Path(exists=True), default=None,
              help="With --label-space copick: inherit each object's model channel from this pretrained "
                   "model_config.yaml (by name) so fine-tuning keeps the pretrained channel binding.")
def cli(config, target, picks_session_id, picks_user_id, seg_target, run_ids,
        tomo_alg, radius_scale, voxel_size,
        target_segmentation_name, target_user_id, target_session_id,
        label_space, from_model):
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
        print(f'    - Pick Targets: {target}')
    elif picks_user_id is not None or picks_session_id is not None:
        print(f'    -  UserID: {picks_user_id} -- SessionID: {picks_session_id}')
    if seg_target is not None and len(seg_target) > 0:
        print(f'    - Segmentation Targets: {seg_target}')
    print()

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
            label_space=label_space,
            from_model=from_model,
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
            label_space=label_space,
            from_model=from_model,
        )


if __name__ == "__main__":
    cli()