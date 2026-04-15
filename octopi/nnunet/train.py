"""
Run nnUNet planning, preprocessing, and training on a converted CoPick dataset.

Expects the dataset to already exist in nnunet_raw (run `octopi nnunet prepare` first).

Steps:
  1. nnUNetv2_plan_and_preprocess  — fingerprint + patch-size planning
  2. nnUNetv2_train                — train one model per requested fold

Supported models (--model flag):
  nnunet              Standard nnUNet               → nnUNetTrainer
  resnecl             Residual Encoder Large        → nnUNetTrainer + nnUNetResEncUNetLPlans
  mednext_s           MedNeXt Small,  kernel 3      → nnUNetTrainerMedNeXtS_kernel3
  mednext_b           MedNeXt Base,   kernel 3      → nnUNetTrainerMedNeXtB_kernel3
  mednext_m           MedNeXt Medium, kernel 3      → nnUNetTrainerMedNeXtM_kernel3
  mednext_l           MedNeXt Large,  kernel 3      → nnUNetTrainerMedNeXtL_kernel3
  mednext_s_k5        MedNeXt Small,  kernel 5      → nnUNetTrainerMedNeXtS_kernel5
  mednext_b_k5        MedNeXt Base,   kernel 5      → nnUNetTrainerMedNeXtB_kernel5
  mednext_m_k5        MedNeXt Medium, kernel 5      → nnUNetTrainerMedNeXtM_kernel5
  mednext_l_k5        MedNeXt Large,  kernel 5      → nnUNetTrainerMedNeXtL_kernel5
"""

import rich_click as click

# MedNeXt trainers require: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
# Trainer classes are defined in octopi/nnunet/mednext_trainer.py and auto-registered
# into nnunetv2's trainer directory on first use.
MODEL_TO_TRAINER = {
    "nnunet":       "nnUNetTrainer",
    "resnecl":      "nnUNetTrainer",
    "mednext_s":    "nnUNetTrainerMedNeXtS_kernel3",
    "mednext_b":    "nnUNetTrainerMedNeXtB_kernel3",
    "mednext_m":    "nnUNetTrainerMedNeXtM_kernel3",
    "mednext_l":    "nnUNetTrainerMedNeXtL_kernel3",
    "mednext_s_k5": "nnUNetTrainerMedNeXtS_kernel5",
    "mednext_b_k5": "nnUNetTrainerMedNeXtB_kernel5",
    "mednext_m_k5": "nnUNetTrainerMedNeXtM_kernel5",
    "mednext_l_k5": "nnUNetTrainerMedNeXtL_kernel5",
}


MEDNEXT_MODELS = {k for k in MODEL_TO_TRAINER if k.startswith("mednext")}


def resolve_trainer(cfg: dict, model_override: str | None) -> tuple[str, str]:
    """
    Return (model_name, trainer_class).

    Priority: --model CLI flag > config.yaml 'model' key > 'nnunet' default.
    """
    from octopi.nnunet.utils import check_mednext_installed
    import sys

    model = model_override or cfg.get("model", "nnunet")
    if model not in MODEL_TO_TRAINER:
        print(f"[ERROR] Unknown model '{model}'. Choose from: {list(MODEL_TO_TRAINER)}")
        sys.exit(1)
    if model in MEDNEXT_MODELS:
        check_mednext_installed()
    return model, MODEL_TO_TRAINER[model]


def set_nnunet_env(cfg: dict) -> dict:
    """Set the three nnUNet path environment variables and return the updated env."""
    from pathlib import Path
    import os

    env = os.environ.copy()
    env["nnUNet_raw"]          = str(cfg["nnunet_raw"])
    env["nnUNet_preprocessed"] = str(cfg["nnunet_preprocessed"])
    env["nnUNet_results"]      = str(cfg["nnunet_results"])

    for key in ("nnunet_preprocessed", "nnunet_results"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)

    return env


def plan_and_preprocess(cfg: dict, env: dict, model: str):
    from octopi.nnunet.utils import _run

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(cfg["dataset_id"]),
        "-c", cfg.get("configuration", "3d_fullres"),
        "--verify_dataset_integrity",
    ]
    if model == "resnecl":
        cmd += ["-pl", "nnUNetPlannerResEncL"]
    _run(cmd, env)


def checkpoint_exists(cfg: dict, trainer: str, model: str, fold: int) -> bool:
    from pathlib import Path

    plans         = "nnUNetResEncUNetLPlans" if model == "resnecl" else "nnUNetPlans"
    configuration = cfg.get("configuration", "3d_fullres")
    dataset_dir   = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    checkpoint    = (
        Path(cfg["nnunet_results"])
        / dataset_dir
        / f"{trainer}__{plans}__{configuration}"
        / f"fold_{fold}"
        / "checkpoint_latest.pth"
    )
    return checkpoint.exists()

def train(cfg: dict, env: dict, model: str, trainer: str, num_gpus: int = 1):
    from octopi.nnunet.utils import _scale_batch_size_for_ddp
    from octopi.nnunet.utils import _run
    import shutil, sys

    dataset_id    = cfg["dataset_id"]
    configuration = cfg.get("configuration", "3d_fullres")
    folds         = cfg.get("folds", [0])

    nnunet_train_bin = shutil.which("nnUNetv2_train")
    if nnunet_train_bin is None:
        print("[ERROR] nnUNetv2_train not found on PATH. Is nnunetv2 installed?")
        sys.exit(1)

    if num_gpus > 1:
        # Scale global batch_size so each GPU sees the same per-GPU sample count
        # as the single-GPU plan (nnUNet DDP requires batch_size >= num_gpus).
        _scale_batch_size_for_ddp(cfg, model, num_gpus)

        # Each DDP worker spawns its own data-augmentation thread pool.
        # Without a cap, 8 workers × N DA threads exhausts CPU RAM (pod OOM).
        # Divide available cores evenly across workers; scheduler env var wins.
        if "nnUNet_n_proc_da" not in env:
            import os
            total_cpu   = os.cpu_count() or 16
            per_gpu_da  = max(2, total_cpu // num_gpus)
            env["nnUNet_n_proc_da"] = str(per_gpu_da)
            print(f"  [DA workers] nnUNet_n_proc_da={per_gpu_da} ({total_cpu} CPUs ÷ {num_gpus} GPUs)")

    print(f"Training with trainer: {trainer}" + (f" on {num_gpus} GPUs" if num_gpus > 1 else ""))
    for fold in folds:
        train_cmd = [
            nnunet_train_bin,
            str(dataset_id),
            configuration,
            str(fold),
            "-tr", trainer,
        ]
        if model == "resnecl":
            train_cmd += ["-p", "nnUNetResEncUNetLPlans"]
        if num_gpus > 1:
            # nnUNetv2_train handles DDP natively via -num_gpus; no torchrun needed.
            train_cmd += ["-num_gpus", str(num_gpus)]
        if checkpoint_exists(cfg, trainer, model, fold):
            print(f"  [fold {fold}] Checkpoint found — resuming.")
            train_cmd += ["--c"]

        _run(train_cmd, env)


@click.command("train", no_args_is_help=True)
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
    help="Model to train (overrides config.yaml). MedNeXt requires nnunet-mednext.",
)
@click.option(
    "--skip-preprocess",
    is_flag=True,
    default=False,
    help="Skip nnUNetv2_plan_and_preprocess (useful if already done).",
)
@click.option(
    "--num-gpus",
    default=1,
    show_default=True,
    type=int,
    help="Number of GPUs for distributed training (uses nnUNet's native -num_gpus flag).",
)
def cli(config, model, skip_preprocess, num_gpus):
    """Plan, preprocess, and train nnUNet on a CoPick dataset."""
    from octopi.nnunet.utils import _load_config

    cfg             = _load_config(config)
    env             = set_nnunet_env(cfg)
    model, trainer  = resolve_trainer(cfg, model)

    if not skip_preprocess:
        plan_and_preprocess(cfg, env, model)

    train(cfg, env, model, trainer, num_gpus=num_gpus)
