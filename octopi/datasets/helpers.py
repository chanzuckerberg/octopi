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

def auto_num_workers(cap: int = 16, min_workers: int = 1, reserve: int = 1) -> int:
    """
    Pick a DataLoader worker count that respects the CPUs actually *allocated*
    to this process.

    Priority order:

    1. SLURM allocation env vars — ``SLURM_CPUS_PER_TASK`` (set when
       ``--cpus-per-task`` is given), falling back to ``SLURM_CPUS_ON_NODE``
       (set for essentially any job step, covering jobs that size CPUs
       differently). These are checked first because on GPU partitions the
       cgroup frequently does *not* restrict CPU affinity, so
       ``os.sched_getaffinity`` reports every core on the node. Trusting that
       would spawn dozens of 100%-CPU workers on a small allocation, starving
       co-tenants and (via copy-on-write cache drift) thrashing node memory
       into swap.
    2. ``os.sched_getaffinity`` — honours cgroup CPU binding when SLURM isn't
       the scheduler (or the vars are unset).
    3. ``os.cpu_count`` — non-Linux fallback.

    ``reserve`` cores are held back for the main process (and the validation
    loader, which sizes itself from this value). The result is clamped to
    ``cap`` as a sanity ceiling: training is loader-bound so throughput keeps
    rising to ~16 workers, but past that gains flatten while RAM/context-switch
    overhead grows, and the cap also guards the auto-detect path from reading a
    whole 64-256 core node as the worker count. The cap does NOT encode a
    per-node "fair share" (e.g. gpu-f = 14 cores/GPU) — that is a scheduling
    policy invisible to this process and must be set via ``--cpus-per-task``.

    The cap only clamps from above: a 4-core laptop still gets ~3 workers.

    ``OCTOPI_MAX_WORKERS`` overrides the count entirely (bypasses cap/reserve)
    for non-SLURM/desktop users on large workstations who want more workers
    than the default cap. SLURM jobs should size via ``--cpus-per-task``
    instead, so this is intended for interactive/local use.
    """
    override = os.environ.get("OCTOPI_MAX_WORKERS")
    if override:
        try:
            return max(min_workers, int(override))
        except ValueError:
            pass  # malformed override -> fall through to auto-detection

    # Prefer SLURM's view of the allocation. Both vars are plain integers;
    # SLURM_JOB_CPUS_PER_NODE is intentionally skipped since it can be a packed
    # form like "12(x2)".
    n = None
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                n = int(val)
                break
            except ValueError:
                continue
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 4  # non-Linux fallback
    return max(min_workers, min(cap, n - reserve))

def parse_resolution_uris(tomo_uris) -> list[tuple[str, float]]:
    """
    Parse tomogram URIs of the form ``alg@voxel_size`` into ``(alg, voxel_size)``
    resolution pairs for multi-source training (mixing voxel sizes and/or reconstruction algorithms).

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

def get_class_info(config, runIDs, target_info, voxel_size) -> tuple[int, list[str]]:
    """
    Get the number of classes and class names from a segmentation.

    Args:
        root: The root of the project.
        runIDs: The list of run IDs to get the class info from.
        target_info: The target information (name, session ID, user ID).
        voxel_size: The voxel size of the segmentation.

    Returns:
        Nclasses: The number of classes.
        class_names: The list of class names.
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
        class_names = target_config['input']['labels']
        Nclasses = len(class_names) + 1
        class_names = [name for name, idx in sorted(class_names.items(), key=lambda x: x[1])]

        # We Only need to read One Segmentation to Get Class Info
        break      

    return Nclasses, class_names

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