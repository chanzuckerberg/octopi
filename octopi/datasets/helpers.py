from octopi.datasets import io as dio
from collections.abc import Mapping
from octopi.utils import io as io
import os
import zarr


def _spatial_shape(obj):
    """Best-effort (z, y, x) shape of a copick tomogram/segmentation, read lazily from its
    zarr store (metadata only, no array load). Returns None if it can't be determined."""
    try:
        z = zarr.open(obj.zarr(), mode="r")
        if hasattr(z, "shape"):            # a zarr array
            shp = z.shape
        elif "0" in z:                     # multiscale group: level 0 = full resolution
            shp = z["0"].shape
        else:
            shp = z[next(iter(z.array_keys()))].shape
        return tuple(int(x) for x in shp[-3:])
    except Exception:
        return None

def auto_num_workers(cap: int = 16, min_workers: int = 1) -> int:
    """
    Pick a DataLoader worker count that respects the CPUs actually available
    to this process.

    On Linux this uses ``os.sched_getaffinity`` so SLURM ``--cpus-per-task``
    bindings are honoured. Elsewhere it falls back to ``os.cpu_count``. The
    result is clamped to ``cap`` since past ~16 workers gains are usually
    noise for a 3D UNet with cached datasets, and each worker adds ~1-2 GB
    of RAM overhead.

    The cap only clamps from above: a 4-core laptop still gets 4 workers.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = 4 # default to small number on non-slurm platforms
    return max(min_workers, min(cap, n))

def parse_resolution_uris(tomo_uris) -> list[tuple[str, float]]:
    """
    Parse tomogram URIs of the form ``alg@voxel_size`` into ``(alg, voxel_size)``
    resolution pairs for multi-resolution training.

    Accepts a single string, a comma-separated string, or a list/tuple of either,
    e.g. ``"wbp@10.0"``, ``"wbp@10.0,wbp@5.0"``, or ``["wbp@10.0", "wbp@5.0"]``.
    Duplicate pairs are removed while preserving first-seen order.
    """
    if isinstance(tomo_uris, str):
        items = [tomo_uris]
    else:
        items = list(tomo_uris)

    resolutions: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for item in items:
        for part in str(item).split(','):
            part = part.strip()
            if not part:
                continue
            if '@' not in part:
                raise ValueError(
                    f"Invalid tomogram URI '{part}': expected 'alg@voxel_size' (e.g. 'wbp@10.0')."
                )
            alg, vs = part.rsplit('@', 1)
            alg = alg.strip()
            try:
                vs = float(vs)
            except ValueError:
                raise ValueError(
                    f"Invalid voxel size in tomogram URI '{part}': '{vs}' is not a number."
                )
            key = (alg, vs)
            if key not in seen:
                seen.add(key)
                resolutions.append(key)

    if not resolutions:
        raise ValueError("No valid tomogram URIs provided (expected 'alg@voxel_size').")
    return resolutions

def scan_runs(
    *,
    root,
    resolutions: list[tuple[str, float]],
    target_name: str,
    target_session_id: str | None,
    target_user_id: str | None,
) -> tuple[dict[str, list[tuple[str, float]]], int, set[tuple[str, float]]]:
    """
    Scan a single CoPick root for the requested (alg, voxel_size) resolutions and
    return only those where BOTH a matching target segmentation and the tomogram
    algorithm exist at that voxel size.

    Args:
        resolutions: list of (alg, voxel_size) pairs to look for.

    Returns:
      - available: {run_id: [(alg, voxel_size), ...]} resolutions present for that run
      - runs_with_seg: number of runs with >=1 matching segmentation (at any requested vs)
      - missing: set of (alg, voxel_size) pairs requested but never found anywhere
    """
    available: dict[str, list[tuple[str, float]]] = {}
    runs_with_seg = 0
    found: set[tuple[str, float]] = set()
    shape_skipped: list[str] = []

    requested = {(a, float(v)) for a, v in resolutions}

    for run in root.runs:
        run_pairs: list[tuple[str, float]] = []
        had_seg = False

        # Cache per-voxel-size lookups so each (run, vs) is queried only once,
        # while still iterating `resolutions` in input order for determinism.
        has_seg_at: dict[float, bool] = {}
        algs_at: dict[float, set[str]] = {}
        seg_shape_at: dict[float, tuple] = {}
        tomo_by_alg_at: dict[float, dict] = {}

        for alg, vs in resolutions:
            vs = float(vs)
            if vs not in has_seg_at:
                seg = run.get_segmentations(
                    name=target_name,
                    session_id=target_session_id,
                    user_id=target_user_id,
                    voxel_size=vs,
                )
                has_seg_at[vs] = len(seg) > 0
                if has_seg_at[vs]:
                    seg_shape_at[vs] = _spatial_shape(seg[0])
                    vs_obj = run.get_voxel_spacing(vs)
                    toms = vs_obj.tomograms if vs_obj is not None else []
                    algs_at[vs] = {t.tomo_type for t in toms}
                    tomo_by_alg_at[vs] = {t.tomo_type: t for t in toms}
                else:
                    algs_at[vs] = set()

            if has_seg_at[vs]:
                had_seg = True
                if alg in algs_at[vs]:
                    # Skip (run, alg) pairs whose tomogram shape disagrees with the target
                    # segmentation shape at this voxel size -- they cannot be cropped together
                    # (mirrors the old prep-multiscale lazy shape check).
                    seg_shp = seg_shape_at.get(vs)
                    tomo_shp = _spatial_shape(tomo_by_alg_at[vs].get(alg))
                    if seg_shp is not None and tomo_shp is not None and seg_shp != tomo_shp:
                        shape_skipped.append(f"{run.name} {alg}@{vs}: tomo {tomo_shp} != seg {seg_shp}")
                        continue
                    run_pairs.append((alg, vs))
                    found.add((alg, vs))

        if had_seg:
            runs_with_seg += 1
        if run_pairs:
            available[run.name] = run_pairs

    if shape_skipped:
        print(
            "\n[Warning] Skipped tomogram/target pairs with mismatched shapes "
            f"({len(shape_skipped)}):\n\t\t" + "\n\t\t".join(shape_skipped) + "\n"
        )

    missing = requested - found
    return available, runs_with_seg, missing

def missing_segmentations(target_name, target_session_id, target_user_id):
    raise RuntimeError(
        f"\n[Error] No segmentations found for the target query:\n"
        f"TargetName: {target_name}, UserID: {target_user_id}, "
        f"SessionID: {target_session_id}\n"
        f"Please check the target name, user ID, and session ID.\n"
    )

def missing_tomograms(missing):
    """
    Warn about requested resolutions that were never found. ``missing`` is a set
    of (alg, voxel_size) pairs.
    """
    pretty = ", ".join(f"{a}@{v}" for a, v in sorted(missing)) if missing else ""
    print(
        f"\n[Warning] The following tomogram/target resolutions are not present in the Copick Project:\n"
        f"\t\t{pretty}\n"
        f"These resolutions will be ignored.\n"
    )

def get_data_splits(
        allRunIDs: dict[str, list[str]],
        trainRunIDs: str = None,
        validateRunIDs: str = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        create_test_dataset: bool = False) -> dict[str, list[str]]:
    """
    Split the available data into training, validation, and testing sets based on input parameters.

    Args:
        trainRunIDs (str): Predefined list of run IDs for training. If provided, it overrides splitting logic.
        validateRunIDs (str): Predefined list of run IDs for validation. If provided with trainRunIDs, no splitting occurs.
        train_ratio (float): Proportion of available data to allocate to the training set.
        val_ratio (float): Proportion of available data to allocate to the validation set.
        test_ratio (float): Proportion of available data to allocate to the test set.
        create_test_dataset (bool): Whether to create a test dataset or leave it empty.

    Returns:
        myRunIDs (dict): Dictionary containing run IDs for training, validation, and testing.
    """          

    # Option 1: Only TrainRunIDs are Provided, Split into Train, Validate and Test (Optional)
    if trainRunIDs is not None and validateRunIDs is None:
        trainRunIDs, validateRunIDs, testRunIDs = dio.split_multiclass_dataset(
            trainRunIDs, train_ratio, val_ratio, test_ratio, 
            return_test_dataset = create_test_dataset
        )
    # Option 2: TrainRunIDs and ValidateRunIDs are Provided, No Need to Split
    elif trainRunIDs is not None and validateRunIDs is not None:
        testRunIDs = []
    # Option 3: Use the Entire Copick Project, Split into Train, Validate and Test
    else:
        runIDs = list(allRunIDs.keys())
        trainRunIDs, validateRunIDs, testRunIDs = dio.split_multiclass_dataset(
            runIDs, train_ratio, val_ratio, test_ratio, 
            return_test_dataset = create_test_dataset
        )

    # Create a map of run IDs to tomogram algorithms
    trainRunIDs = {rid: allRunIDs[rid] for rid in trainRunIDs if rid in allRunIDs}
    validateRunIDs  = {rid: allRunIDs[rid] for rid in validateRunIDs if rid in allRunIDs}
    testRunIDs = {rid: allRunIDs[rid] for rid in testRunIDs if rid in allRunIDs}

    # Swap if Test Runs is Larger than Validation Runs
    if create_test_dataset and len(testRunIDs) > len(validateRunIDs):
        testRunIDs, validateRunIDs = validateRunIDs, testRunIDs

    # Store the split run IDs into a dictionary for easy access
    myRunIDs = {
        'train': trainRunIDs,
        'validate': validateRunIDs,
        'test': testRunIDs
    }  

    return myRunIDs

def get_class_info(config, runIDs, target_info, voxel_size) -> tuple[int, list[str], str, dict]:
    """
    Get the number of classes and class names from a segmentation.

    Args:
        root: The root of the project.
        runIDs: The list of run IDs to get the class info from.
        target_info: The target information (name, session ID, user ID).
        voxel_size: The voxel size of the segmentation.

    Returns:
        Nclasses: The number of classes (foreground objects + background).
        class_names: The object names sorted by ascending recorded label (model channel).
        label_space: 'copick' when the persisted seg holds copick global labels, else None.
        model_labels: The recorded {name: model_label} map from the targets YAML.
    """

    # Load the Copick Config
    root = dio.load_copick_config(config)

    # Fetch a segmentation to determine class names and number of classes
    target_name, target_session_id, target_user_id = target_info
    for runID in runIDs:
        run = root.get_run(runID)
        seg = run.get_segmentations(name=target_name, 
                                    session_id=target_session_id, 
                                    user_id=target_user_id,
                                    voxel_size=float(voxel_size))
        if len(seg) == 0:
            continue

        # If Session ID or User ID are None, Set Them Based on the First Found Segmentation
        if target_session_id is None:
            target_session_id = seg[0].session_id
        if target_user_id is None:
            target_user_id = seg[0].user_id

        # Read Yaml Config to Get Number of Classes and Class Names
        target_config = io.get_config(
            config, target_name, 'targets', 
            target_user_id, target_session_id
        )
        # `labels` maps name -> recorded label (model channel in copick space, else the
        # legacy sequential value). `label_space` marks whether the seg holds copick globals.
        model_labels = target_config['input']['labels']
        label_space = target_config['input'].get('label_space')
        Nclasses = len(model_labels) + 1
        # Order object names by their model channel so class_names[i] aligns to channel i+1.
        class_names = [name for name, _ in sorted(model_labels.items(), key=lambda kv: kv[1])]

        # We Only need to read One Segmentation to Get Class Info
        break

    return Nclasses, class_names, label_space, model_labels

def build_compaction_map(root, model_labels, label_space):
    """
    Build the (orig_labels, target_labels) pair for MONAI's MapLabelValued that compacts the
    persisted copick *global* labels into the model's dense channels, for ``label_space ==
    'copick'`` targets. The mapping is read explicitly from the recorded ``model_labels``
    ({name: model_label}) joined with the project config's global labels by name — no sorting.

    Legacy targets (``label_space`` not 'copick') are already in model/dense space, so no remap
    is needed and ``(None, None)`` is returned (``get_transforms`` then skips MapLabelValued).

    In copick mode every project object's global label maps to its model channel; objects that
    are painted but not selected as training targets fold to background 0 (safe — copick global
    labels are unique). Background maps to background.

    Args:
        root: A copick root for the (training) project the labels are painted in.
        model_labels: The recorded {name: model_label} map (from get_class_info).
        label_space: 'copick' to build the map, otherwise legacy no-op.

    Returns:
        (orig_labels, target_labels) for ``MapLabelValued``, or (None, None) for legacy.
    """
    if label_space != 'copick':
        return None, None
    orig_labels = [0] + [o.label for o in root.pickable_objects]
    target_labels = [0] + [model_labels.get(o.name, 0) for o in root.pickable_objects]
    return orig_labels, target_labels

def build_target_uri(name: str, sessionid: str | None, userid: str | None, voxel_size: float) -> str:
    """
    Build the target URI from the target information.

    Args:
        name: The name of the target.
        sessionid: The session ID of the target.
        userid: The user ID of the target.
        voxel_size: The voxel size of the target.
    """
    # Construct the Target URI
    if sessionid is None and userid is None:
        uri = f'{name}@{voxel_size}'
    elif sessionid is None:
        uri = f'{name}:{userid}@{voxel_size}'
    else:
        uri = f'{name}:{userid}/{sessionid}@{voxel_size}'

    return uri

def print_splits(myRunIDs, train_files, val_files):
    """
    Print the data splits.
    """
    total_train = sum(len(d) for d in myRunIDs["train"].values())
    total_val = sum(len(d) for d in myRunIDs["validate"].values())
    total_test = sum(len(d) for d in myRunIDs["test"].values())

    print('\n🔍 Data Splits:')
    print(f'# training -- Runs={total_train}, Tomograms={len(train_files)}')
    print(f'# validation -- Runs={total_val}, Tomograms={len(val_files)}')
    print(f'# test runs={total_test}\n')

def check_max_label_value(Nclasses, train_files):
    max_label_value = max(file['label'].max() for file in train_files)
    if max_label_value > Nclasses:
        print(f"Warning: Maximum class label value {max_label_value} exceeds the number of classes {Nclasses}.")
        print("This may cause issues with the model's output layer.")
        print("Consider adjusting the number of classes or the label values in your data.\n")

def get_parameters(datamodule):
    """
    Return datamodule parameters in a format that depends on whether
    this is a single-config or multi-config datamodule.
    """

    # Multi-resolution: a list of (alg, voxel_size) pairs. Record the full set
    # of training URIs plus representative scalars (first resolution) for any
    # downstream consumer that still expects a single voxel_size / algorithm.
    resolutions = datamodule.resolutions
    tomo_uris = [f"{alg}@{vs}" for alg, vs in resolutions]
    unique_vss = sorted({vs for _, vs in resolutions})

    base = {
        "target_info": [datamodule.target_user_id, datamodule.target_session_id, datamodule.target_name],
        "tomo_uris": tomo_uris,
        "target_uris": [
            build_target_uri(datamodule.target_name, datamodule.target_session_id, datamodule.target_user_id, vs)
            for vs in unique_vss
        ],
        # Representative scalars (first resolution) for backward compatibility.
        "voxel_size": resolutions[0][1],
        "tomo_algorithm": sorted({alg for alg, _ in resolutions}),
        "background_ratio": datamodule.bgr,
    }

    # -------------------------
    # SINGLE-CONFIG
    # -------------------------
    if isinstance(datamodule.config, str):
        return {
            **base,
            "config": datamodule.config,
            "trainRunIDs": list(datamodule.myRunIDs["train"].keys()),
            "valRunIDs": list(datamodule.myRunIDs["validate"].keys()),
        }

    # -------------------------
    # MULTI-CONFIG
    # -------------------------
    if isinstance(datamodule.config, Mapping):
        configs_out = []

        for session_key, config_path in datamodule.config.items():
            configs_out.append({
                "session": session_key,
                "config": config_path,
                "trainRunIDs": list(
                    datamodule.myRunIDs["train"].get(session_key, {}).keys()
                ),
                "valRunIDs": list(
                    datamodule.myRunIDs["validate"].get(session_key, {}).keys()
                ),
            })

        return {
            **base,
            "configs": configs_out,
        }

    # -------------------------
    # FALLBACK
    # -------------------------
    raise TypeError(
        f"Unsupported type for datamodule.config: {type(datamodule.config)}"
    )    