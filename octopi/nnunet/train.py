"""
Run nnUNet planning, preprocessing, and training on a converted CoPick dataset.

Expects the dataset to already exist in nnunet_raw (run `octopi nnunet prepare` first).

Steps:
  1. nnUNetv2_plan_and_preprocess  — fingerprint + patch-size planning
  2. nnUNetv2_train                — train one model per requested fold

Supported models (--model flag):
  nnunet              Standard nnUNet               → nnUNetTrainer
  mednext_s           MedNeXt Small,  kernel 3      → nnUNetTrainerMedNeXtS_kernel3
  mednext_b           MedNeXt Base,   kernel 3      → nnUNetTrainerMedNeXtB_kernel3
  mednext_m           MedNeXt Medium, kernel 3      → nnUNetTrainerMedNeXtM_kernel3
  mednext_l           MedNeXt Large,  kernel 3      → nnUNetTrainerMedNeXtL_kernel3
  mednext_s_k5        MedNeXt Small,  kernel 5      → nnUNetTrainerMedNeXtS_kernel5
  mednext_b_k5        MedNeXt Base,   kernel 5      → nnUNetTrainerMedNeXtB_kernel5
  mednext_m_k5        MedNeXt Medium, kernel 5      → nnUNetTrainerMedNeXtM_kernel5
  mednext_l_k5        MedNeXt Large,  kernel 5      → nnUNetTrainerMedNeXtL_kernel5
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml
import rich_click as click

# Mapping from friendly model name → nnUNet trainer class.
# MedNeXt trainers are provided by the nnunet-mednext package:
#   pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
MODEL_TO_TRAINER = {
    "nnunet":       "nnUNetTrainer",
    "mednext_s":    "nnUNetTrainerMedNeXtS_kernel3",
    "mednext_b":    "nnUNetTrainerMedNeXtB_kernel3",
    "mednext_m":    "nnUNetTrainerMedNeXtM_kernel3",
    "mednext_l":    "nnUNetTrainerMedNeXtL_kernel3",
    "mednext_s_k5": "nnUNetTrainerMedNeXtS_kernel5",
    "mednext_b_k5": "nnUNetTrainerMedNeXtB_kernel5",
    "mednext_m_k5": "nnUNetTrainerMedNeXtM_kernel5",
    "mednext_l_k5": "nnUNetTrainerMedNeXtL_kernel5",
}


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_trainer(cfg: dict, model_override: str | None) -> str:
    """
    Determine the nnUNet trainer class to use.

    Priority: --model CLI flag > config.yaml 'model' key > config.yaml 'trainer' key > 'nnunet' default.
    """
    model = model_override or cfg.get("model")
    if model:
        if model not in MODEL_TO_TRAINER:
            print(f"[ERROR] Unknown model '{model}'. Choose from: {list(MODEL_TO_TRAINER)}")
            sys.exit(1)
        return MODEL_TO_TRAINER[model]
    return cfg.get("trainer", "nnUNetTrainer")


def set_nnunet_env(cfg: dict) -> dict:
    """Set the three nnUNet path environment variables and return the updated env."""
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(cfg["nnunet_raw"])
    env["nnUNet_preprocessed"] = str(cfg["nnunet_preprocessed"])
    env["nnUNet_results"]      = str(cfg["nnunet_results"])

    for key in ("nnunet_preprocessed", "nnunet_results"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)

    return env


def _run(cmd: list[str], env: dict):
    """Run a subprocess command, streaming output, and exit on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def plan_and_preprocess(cfg: dict, env: dict):
    _run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(cfg["dataset_id"]),
        "-c", cfg.get("configuration", "3d_fullres"),
        "--verify_dataset_integrity",
    ], env)


def train(cfg: dict, env: dict, trainer: str):
    dataset_id    = cfg["dataset_id"]
    configuration = cfg.get("configuration", "3d_fullres")
    folds         = cfg.get("folds", [0])

    print(f"Training with trainer: {trainer}")
    for fold in folds:
        _run([
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "--trainer", trainer,
        ], env)


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
def cli(config, model, skip_preprocess):
    """Plan, preprocess, and train nnUNet on a CoPick dataset."""
    cfg     = _load_config(config)
    env     = set_nnunet_env(cfg)
    trainer = resolve_trainer(cfg, model)

    if not skip_preprocess:
        plan_and_preprocess(cfg, env)

    train(cfg, env, trainer)
