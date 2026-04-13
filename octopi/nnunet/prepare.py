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

import json
from pathlib import Path

import copick
import numpy as np
import SimpleITK as sitk
import yaml
import rich_click as click
from copick.util.uri import resolve_copick_objects
from tqdm import tqdm

from octopi.datasets.helpers import build_target_uri
from octopi.utils.io import get_config


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_to_case_id(run_name: str) -> str:
    """Sanitize a CoPick run name into a valid nnUNet case identifier."""
    return run_name.replace("-", "_").replace(" ", "_")


def array_to_nifti(data: np.ndarray, voxel_size_angstrom: float) -> sitk.Image:
    """
    Wrap a (Z, Y, X) numpy array in a SimpleITK image.

    Spacing is converted from Angstroms to nanometres (divide by 10) so that
    nnUNet's patch-size planner sees reasonable numbers.
    """
    spacing_nm = float(voxel_size_angstrom) / 10.0
    img = sitk.GetImageFromArray(data)
    img.SetSpacing([spacing_nm, spacing_nm, spacing_nm])  # SimpleITK uses (x, y, z)
    return img


def get_label_map(copick_config: str, seg_name: str, user_id: str, session_id: str) -> dict:
    """
    Return {class_name: integer_label} from the OCTOPI targets config stored
    in the CoPick overlay.  Background (0) is added automatically.
    """
    target_cfg = get_config(copick_config, seg_name, "targets", user_id, session_id)
    labels = {"background": 0}
    for name, idx in target_cfg["input"]["labels"].items():
        labels[name] = idx
    return labels


def load_volume(root, vol_uri: str, run_name: str) -> np.ndarray:
    vols = resolve_copick_objects(vol_uri, root, "tomogram", run_name=run_name)
    if not vols:
        raise RuntimeError(f"No tomogram found for run '{run_name}' with URI '{vol_uri}'")
    return vols[0].numpy()


def load_segmentation(root, seg_uri: str, run_name: str) -> np.ndarray:
    segs = resolve_copick_objects(seg_uri, root, "segmentation", run_name=run_name)
    if not segs:
        raise RuntimeError(f"No segmentation found for run '{run_name}' with URI '{seg_uri}'")
    return segs[0].numpy().astype(np.uint8)


def convert(cfg: dict):
    copick_cfg   = cfg["copick_config"]
    tomo_alg     = cfg["tomo_algorithm"]
    voxel_size   = cfg["voxel_size"]
    seg_name     = cfg["segmentation_name"]
    user_id      = cfg.get("segmentation_user_id", "octopi")
    session_id   = cfg.get("segmentation_session_id", "1")

    train_run_ids = cfg.get("train_run_ids") or []
    test_run_ids  = cfg.get("test_run_ids")  or []

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
    print(f"Converting {len(train_run_ids)} training runs...")
    for run_name in tqdm(train_run_ids):
        case_id = run_to_case_id(run_name)
        try:
            tomo_data = load_volume(root, vol_uri, run_name)
            seg_data  = load_segmentation(root, seg_uri, run_name)
        except RuntimeError as e:
            print(f"  [SKIP] {e}")
            skipped.append(run_name)
            continue

        sitk.WriteImage(
            array_to_nifti(tomo_data.astype(np.float32), voxel_size),
            str(images_tr / f"{case_id}_0000.nii.gz"),
        )
        sitk.WriteImage(
            array_to_nifti(seg_data, voxel_size),
            str(labels_tr / f"{case_id}.nii.gz"),
        )
        n_training += 1

    # Test cases (images only — no labels written)
    if test_run_ids:
        print(f"\nConverting {len(test_run_ids)} test runs...")
        for run_name in tqdm(test_run_ids):
            case_id = run_to_case_id(run_name)
            try:
                tomo_data = load_volume(root, vol_uri, run_name)
            except RuntimeError as e:
                print(f"  [SKIP] {e}")
                continue
            sitk.WriteImage(
                array_to_nifti(tomo_data.astype(np.float32), voxel_size),
                str(images_ts / f"{case_id}_0000.nii.gz"),
            )

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "cryo-ET"},
        "labels": labels_dict,
        "numTraining": n_training,
        "file_endings": ".nii.gz",
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
@click.option(
    "-c", "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to nnunet config.yaml",
)
def cli(config):
    """Convert a CoPick project to nnUNet raw dataset format (imagesTr / labelsTr / imagesTs)."""
    convert(_load_config(config))
