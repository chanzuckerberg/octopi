"""
Generate a boilerplate nnUNet config YAML for use with octopi nnunet commands.
"""
import rich_click as click

BOILERPLATE = """\
# ── CoPick project ────────────────────────────────────────────────────────────
copick_config: /path/to/copick_config.json

# ── Tomogram / segmentation identifiers ──────────────────────────────────────
tomo_algorithm: wbp            # e.g. wbp, denoised, ctfdeconv
voxel_size: 10.0               # Angstroms
segmentation_name: paintedPicks
segmentation_user_id: octopi
segmentation_session_id: "1"

# ── Run selection ─────────────────────────────────────────────────────────────
# Leave train_run_ids empty ([]) to use all runs not listed in test_run_ids.
train_run_ids: []
test_run_ids: []

# ── nnUNet dataset ────────────────────────────────────────────────────────────
dataset_id: 1                  # Integer; becomes Dataset001_<dataset_name>
dataset_name: MyDataset

# ── nnUNet paths ──────────────────────────────────────────────────────────────
nnunet_raw: /path/to/nnunet_raw
nnunet_preprocessed: /path/to/nnunet_preprocessed
nnunet_results: /path/to/nnunet_results

# ── Training ──────────────────────────────────────────────────────────────────
configuration: 3d_fullres      # 3d_fullres | 3d_lowres | 3d_cascade_fullres
folds: [0]                     # [0,1,2,3,4] for full 5-fold cross-validation
model: nnunet                  # nnunet | mednext_s | mednext_b | mednext_m | mednext_l
                               # Append _k5 for kernel-5 MedNeXt variants

# ── Inference ─────────────────────────────────────────────────────────────────
predictions_dir: /path/to/predictions

# ── Prepare options ───────────────────────────────────────────────────────────
num_workers: 4                 # Parallel threads for writing .nii.gz files (prepare)
"""


@click.command("config", no_args_is_help=False)
@click.argument(
    "output",
    default="nnunet_config.yaml",
    metavar="OUTPUT",
    required=False,
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Overwrite OUTPUT if it already exists.",
)
def cli(output, force):
    """Write a boilerplate nnUNet config YAML to OUTPUT (default: nnunet_config.yaml)."""
    import sys
    from pathlib import Path

    out = Path(output)
    if out.exists() and not force:
        print(f"[ERROR] '{out}' already exists. Use --force to overwrite.")
        sys.exit(1)

    out.write_text(BOILERPLATE)
    print(f"Boilerplate config written to: {out}")
    print("Edit the file, then run:")
    print(f"  octopi nnunet prepare -c {out}")
    print(f"  octopi nnunet train   -c {out}")
    print(f"  octopi nnunet predict -c {out}")
