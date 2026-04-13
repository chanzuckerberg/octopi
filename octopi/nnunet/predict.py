"""
Run nnUNet inference on test tomograms and optionally write predictions
back into the CoPick project as segmentations.

Expects:
  - imagesTs/ to be populated (run `octopi nnunet prepare` with test_run_ids set)
  - A trained model in nnunet_results (run `octopi nnunet train` first)
"""

from octopi.nnunet.train import MODEL_TO_TRAINER, resolve_trainer, set_nnunet_env
import rich_click as click


def run_inference(cfg: dict, env: dict, trainer: str):
    from pathlib import Path
    import sys

    dataset_id      = cfg["dataset_id"]
    dataset_name    = cfg["dataset_name"]
    configuration   = cfg.get("configuration", "3d_fullres")
    folds           = cfg.get("folds", [0])
    predictions_dir = Path(cfg["predictions_dir"])
    nnunet_raw      = Path(cfg["nnunet_raw"])

    predictions_dir.mkdir(parents=True, exist_ok=True)

    images_ts = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesTs"
    if not images_ts.exists() or not list(images_ts.glob("*.nii.gz")):
        print(f"[ERROR] No test images found in {images_ts}")
        print("  Run `octopi nnunet prepare` with test_run_ids set first.")
        sys.exit(1)

    fold_args = []
    for f in folds:
        fold_args += ["-f", str(f)]

    _run([
        "nnUNetv2_predict",
        "-i", str(images_ts),
        "-o", str(predictions_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-tr", trainer,
        *fold_args,
    ], env)


def _save_to_copick(cfg: dict):

    import copick, sys, SimpleITK as sitk, numpy as np
    from copick_utils.io import writers
    from pathlib import Path
    from tqdm import tqdm

    """
    Read predicted .nii.gz files from predictions_dir and write them
    back into the CoPick project as segmentations under user_id='nnunet'.
    """
    copick_cfg      = cfg["copick_config"]
    voxel_size      = cfg["voxel_size"]
    seg_name        = cfg["segmentation_name"]
    predictions_dir = Path(cfg["predictions_dir"])

    root = copick.from_file(copick_cfg)

    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"[ERROR] No prediction files found in {predictions_dir}")
        sys.exit(1)

    print(f"\nWriting {len(pred_files)} predictions back to CoPick...")
    for pred_path in tqdm(pred_files):
        # Recover run name from case ID (reverse the sanitization in prepare.py)
        case_id  = pred_path.stem.replace(".nii", "")  # strip .nii from .nii.gz stem
        run_name = case_id.replace("_", "-")            # best-effort; adjust if needed

        run = root.get_run(run_name)
        if run is None:
            print(f"  [SKIP] No CoPick run found for case '{case_id}' (tried '{run_name}')")
            continue

        img  = sitk.ReadImage(str(pred_path))
        pred = sitk.GetArrayFromImage(img).astype(np.uint8)  # (Z, Y, X)

        writers.segmentation(
            run, pred,
            voxel_size=voxel_size,
            name=seg_name,
            user_id="nnunet",
            session_id="0",
        )

    print("Done writing predictions to CoPick.")


@click.command("predict", no_args_is_help=True)
@click.option(
    "-c", "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to nnunet config.yaml",
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_TO_TRAINER)),
    default=None,
    help="Model to predict with — must match what was used during training.",
)
@click.option(
    "--save-to-copick",
    is_flag=True,
    default=False,
    help="Write predictions back into the CoPick project as segmentations (user_id='nnunet').",
)
def cli(config, model, save_to_copick):
    from octopi.nnunet.utils import _load_config

    """Run nnUNet inference on CoPick test tomograms."""
    cfg     = _load_config(config)
    env     = set_nnunet_env(cfg)
    trainer = resolve_trainer(cfg, model)

    run_inference(cfg, env, trainer)

    if save_to_copick:
        _save_to_copick(cfg)
