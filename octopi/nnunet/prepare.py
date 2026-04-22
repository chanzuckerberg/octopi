"""
Convert a CoPick project to nnUNet raw dataset format.

Reads tomograms and segmentation masks from CoPick and writes them as
.nii.gz files in the nnUNet Dataset folder structure:

    nnunet_raw/
    └── Dataset{id}_{name}/
        ├── dataset.json
        ├── imagesTr/   {case}_0000.nii.gz
        ├── labelsTr/   {case}.nii.gz
        └── imagesTs/   {case}_0000.nii.gz   (if test_run_ids provided)
"""
import rich_click as click
import numpy as np
from octopi.utils import parsers

def run_to_case_id(run_name: str) -> str:
    """Sanitize a CoPick run name into a valid nnUNet case identifier."""
    return run_name.replace("-", "_").replace(" ", "_")


def array_to_nifti(data: np.ndarray, voxel_size_angstrom: float):
    """
    Wrap a (Z, Y, X) numpy array in a SimpleITK image.

    Spacing is converted from Angstroms to nanometres (divide by 10) so that
    nnUNet's patch-size planner sees reasonable numbers.
    """
    import SimpleITK as sitk

    spacing_nm = float(voxel_size_angstrom) / 10.0
    img = sitk.GetImageFromArray(data)
    img.SetSpacing([spacing_nm, spacing_nm, spacing_nm])  # SimpleITK uses (x, y, z)
    return img


def get_label_map(copick_config: str, seg_name: str, user_id: str, session_id: str) -> dict:
    """
    Return {class_name: integer_label} from the OCTOPI targets config stored
    in the CoPick overlay.  Background (0) is added automatically.
    """
    from octopi.utils.io import get_config

    target_cfg = get_config(copick_config, seg_name, "targets", user_id, session_id)
    labels = {"background": 0}
    for name, idx in target_cfg["input"]["labels"].items():
        labels[name] = idx
    return labels


def load_volume(root, vol_uri: str, run_name: str) -> np.ndarray:
    from copick.util.uri import resolve_copick_objects
    
    vols = resolve_copick_objects(vol_uri, root, "tomogram", run_name=run_name)
    if not vols:
        raise RuntimeError(f"No tomogram found for run '{run_name}' with URI '{vol_uri}'")
    return vols[0].numpy()


def load_segmentation(root, seg_uri: str, run_name: str) -> np.ndarray:
    from copick.util.uri import resolve_copick_objects

    segs = resolve_copick_objects(seg_uri, root, "segmentation", run_name=run_name)
    if not segs:
        raise RuntimeError(f"No segmentation found for run '{run_name}' with URI '{seg_uri}'")
    return segs[0].numpy().astype(np.uint8)


def _process_train_run(args):
    """Worker: load one training tomogram + segmentation and write nii.gz files."""
    run_name, root, vol_uri, seg_uri, voxel_size, images_tr, labels_tr = args
    import SimpleITK as sitk

    case_id = run_to_case_id(run_name)
    try:
        tomo_data = load_volume(root, vol_uri, run_name)
        seg_data  = load_segmentation(root, seg_uri, run_name)
    except RuntimeError as e:
        return run_name, False, str(e)

    sitk.WriteImage(
        array_to_nifti(tomo_data.astype(np.float32), voxel_size),
        str(images_tr / f"{case_id}_0000.nii.gz"),
    )
    sitk.WriteImage(
        array_to_nifti(seg_data, voxel_size),
        str(labels_tr / f"{case_id}.nii.gz"),
    )
    return run_name, True, None


def _process_test_run(args):
    """Worker: load one test tomogram and write nii.gz file."""
    run_name, root, vol_uri, voxel_size, images_ts = args
    import SimpleITK as sitk

    case_id = run_to_case_id(run_name)
    try:
        tomo_data = load_volume(root, vol_uri, run_name)
    except RuntimeError as e:
        return run_name, False, str(e)

    sitk.WriteImage(
        array_to_nifti(tomo_data.astype(np.float32), voxel_size),
        str(images_ts / f"{case_id}_0000.nii.gz"),
    )
    return run_name, True, None


def convert(cfg: dict):
    from octopi.datasets.helpers import build_target_uri
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from tqdm import tqdm
    import copick
    import json

    copick_cfg   = cfg["copick_config"]
    tomo_alg     = cfg["tomo_algorithm"]
    voxel_size   = cfg["voxel_size"]
    seg_name     = cfg["segmentation_name"]
    user_id      = cfg.get("segmentation_user_id", "octopi")
    session_id   = cfg.get("segmentation_session_id", "1")
    num_workers  = cfg.get("num_workers", 4)

    train_run_ids = [str(r) for r in (cfg.get("train_run_ids") or [])]
    test_run_ids  = [str(r) for r in (cfg.get("test_run_ids")  or [])]

    dataset_id   = cfg["dataset_id"]
    dataset_name = cfg["dataset_name"]
    nnunet_raw   = Path(cfg["nnunet_raw"])

    vol_uri = f"{tomo_alg}@{voxel_size}"
    seg_uri = build_target_uri(seg_name, session_id, user_id, voxel_size)

    dataset_dir = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr   = dataset_dir / "imagesTr"
    labels_tr   = dataset_dir / "labelsTr"
    images_ts   = dataset_dir / "imagesTs"

    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    root = copick.from_file(copick_cfg)
    all_runs = [r.name for r in root.runs]

    if not train_run_ids:
        train_run_ids = [r for r in all_runs if r not in test_run_ids]

    labels_dict = get_label_map(copick_cfg, seg_name, user_id, session_id)

    # Training cases
    n_training = 0
    skipped    = []
    print(f"Attempting to convert {len(train_run_ids)} training runs "
          f"(tomo={vol_uri}, seg={seg_uri}, num_workers={num_workers})...")
    train_args = [
        (run_name, root, vol_uri, seg_uri, voxel_size, images_tr, labels_tr)
        for run_name in train_run_ids
    ]
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_train_run, a): a[0] for a in train_args}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            run_name, ok, err = fut.result()
            if ok:
                n_training += 1
            else:
                print(f"  [SKIP] {err}")
                skipped.append(run_name)

    if n_training == 0:
        import sys
        print(f"\n[ERROR] No training cases were written. "
              f"Check that runs have both a tomogram matching '{vol_uri}' "
              f"and a segmentation matching '{seg_uri}'.")
        sys.exit(1)

    # Test cases (images only — no labels written)
    if test_run_ids:
        print(f"\nConverting {len(test_run_ids)} test runs (num_workers={num_workers})...")
        test_args = [
            (run_name, root, vol_uri, voxel_size, images_ts)
            for run_name in test_run_ids
        ]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_test_run, a): a[0] for a in test_args}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                run_name, ok, err = fut.result()
                if not ok:
                    print(f"  [SKIP] {err}")

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "cryo-ET"},
        "labels": labels_dict,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nDone. Dataset written to: {dataset_dir}")
    print(f"  Training cases : {n_training}  (skipped: {len(skipped)})")
    print(f"  Test cases     : {len(test_run_ids)}")
    print(f"  Labels         : {labels_dict}")
    if skipped:
        print(f"  Skipped runs   : {skipped}")


@click.command("prepare", no_args_is_help=True)
# Input Arguments
@click.option("-c", "--config", required=True, type=click.Path(exists=True),
              help="Path to copick config.json")
@click.option("-vs", "--voxel-size", type=float, default=10.0, show_default=True,
              help="Voxel size of tomograms in Angstroms")
@click.option("-alg", "--tomo-alg", type=str, default="wbp", show_default=True,
              help="Tomogram algorithm (e.g. wbp, denoised)")
@click.option("-seginfo", "--seg-info", type=str, default="paintedPicks,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value) if value else value,
              help="Segmentation info as 'name,user_id,session_id'")
@click.option("-truns", "--train-run-ids", type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="Training run IDs, e.g. run1,run2,run3. Default: all runs not in test set.")
@click.option("-tests", "--test-run-ids", type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="Test run IDs, e.g. run4,run5")
# Output Arguments
@click.option("-did", "--dataset-id", type=int, required=True,
              help="nnUNet dataset ID (integer; becomes Dataset{id}_{name})")
@click.option("-dname", "--dataset-name", type=str, required=True,
              help="nnUNet dataset name")
@click.option("-raw", "--raw", "nnunet_raw", type=click.Path(), required=True,
              help="Path to nnunet_raw output directory")
@click.option("-j", "--num-workers", default=4, show_default=True, type=int,
              help="Number of parallel worker threads for converting tomograms.")
def cli(config, voxel_size, tomo_alg, seg_info, train_run_ids, test_run_ids,
        dataset_id, dataset_name, nnunet_raw, num_workers):
    """Convert a CoPick project to nnUNet raw dataset format (imagesTr / labelsTr / imagesTs)."""
    seg_name, user_id, session_id = seg_info
    if user_id is None:
        user_id = "octopi"
    if session_id is None:
        session_id = "1"

    cfg = {
        "copick_config":          config,
        "tomo_algorithm":         tomo_alg,
        "voxel_size":             voxel_size,
        "segmentation_name":      seg_name,
        "segmentation_user_id":   user_id,
        "segmentation_session_id": session_id,
        "dataset_id":             dataset_id,
        "dataset_name":           dataset_name,
        "nnunet_raw":             nnunet_raw,
        "train_run_ids":          train_run_ids or [],
        "test_run_ids":           test_run_ids or [],
        "num_workers":            num_workers,
    }
    convert(cfg)
