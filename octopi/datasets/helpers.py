from octopi.datasets import io as dio
from collections.abc import Mapping
from octopi.utils import io as io

def scan_runs(
    *,
    root,
    voxel_size: float,
    target_name: str,
    target_session_id: str | None,
    target_user_id: str | None,
    requested_algs: set[str],
) -> tuple[dict[str, list[str]], int, set[str]]:
    """
    Scan a single CoPick root and return:
      - available: {run_id: [matched_algs]}
      - runs_with_seg: number of runs that had matching segmentation
      - algs_present: union of matched algs across available runs (for warnings)
    """
    available: dict[str, list[str]] = {}
    runs_with_seg = 0

    runIDs = [run.name for run in root.runs]
    for runID in runIDs:
        run = root.get_run(runID)

        seg = run.get_segmentations(
            name=target_name,
            session_id=target_session_id,
            user_id=target_user_id,
            voxel_size=float(voxel_size),
        )
        if len(seg) == 0:
            continue
        runs_with_seg += 1

        tomos = run.get_voxel_spacing(voxel_size).tomograms
        run_algs = {t.tomo_type for t in tomos}

        matched = sorted(requested_algs & run_algs) if requested_algs else sorted(run_algs)
        if matched:
            available[runID] = matched

    algs_present = set().union(*available.values()) if available else set()
    return available, runs_with_seg, algs_present

def missing_segmentations(target_name, target_session_id, target_user_id):
    raise RuntimeError(
        f"\n[Error] No segmentations found for the target query:\n"
        f"TargetName: {target_name}, UserID: {target_user_id}, "
        f"SessionID: {target_session_id}\n"
        f"Please check the target name, user ID, and session ID.\n"
    )

def missing_tomograms(algorithms):
    print(
        f"\n[Warning] The following tomogram algorithms are not present in the Copick Project:\n"
        f"\t\t{algorithms}\n"
        f"These tomogram algorithms will be ignored.\n"
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

    base = {
        "target_uri": build_target_uri(datamodule.target_name, datamodule.target_session_id, datamodule.target_user_id, datamodule.voxel_size),
        "target_info": [datamodule.target_user_id, datamodule.target_session_id, datamodule.target_name],
        "voxel_size": datamodule.voxel_size,
        "tomo_algorithm": datamodule.tomo_alg,
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