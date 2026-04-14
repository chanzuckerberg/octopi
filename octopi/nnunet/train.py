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
MEDNEXT_INSTALL = "pip install git+https://github.com/MIC-DKFZ/MedNeXt.git"


def check_mednext_installed():
    try:
        import importlib
        importlib.import_module("nnunetv2.training.nnUNetTrainer.variants.MedNeXt.nnUNetTrainerMedNeXt")
    except ModuleNotFoundError:
        import sys
        print("[ERROR] MedNeXt is not installed. Run:")
        print(f"  {MEDNEXT_INSTALL}")
        sys.exit(1)


def resolve_trainer(cfg: dict, model_override: str | None) -> tuple[str, str]:
    """
    Return (model_name, trainer_class).

    Priority: --model CLI flag > config.yaml 'model' key > 'nnunet' default.
    """
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


def train(cfg: dict, env: dict, model: str, trainer: str, num_gpus: int = 1, num_epochs: int | None = None):
    from octopi.nnunet.utils import _run

    dataset_id    = cfg["dataset_id"]
    configuration = cfg.get("configuration", "3d_fullres")
    folds         = cfg.get("folds", [0])

    print(f"Training with trainer: {trainer}" + (f" on {num_gpus} GPUs" if num_gpus > 1 else ""))
    for fold in folds:
        train_cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "-tr", trainer,
        ]
        if model == "resnecl":
            train_cmd += ["-p", "nnUNetResEncUNetLPlans"]
        if num_epochs is not None:
            train_cmd += ["-num_epochs", str(num_epochs)]
        if checkpoint_exists(cfg, trainer, model, fold):
            print(f"  [fold {fold}] Checkpoint found — resuming.")
            train_cmd += ["--c"]

        if num_gpus > 1:
            cmd = ["torchrun", f"--nproc_per_node={num_gpus}"] + train_cmd
        else:
            cmd = train_cmd
        _run(cmd, env)


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
    help="Number of GPUs for distributed training via torchrun.",
)
@click.option(
    "--num-epochs",
    default=None,
    type=int,
    help="Override number of training epochs (default: nnUNet's 1000).",
)
def cli(config, model, skip_preprocess, num_gpus, num_epochs):
    """Plan, preprocess, and train nnUNet on a CoPick dataset."""
    from octopi.nnunet.utils import _load_config

    cfg             = _load_config(config)
    env             = set_nnunet_env(cfg)
    model, trainer  = resolve_trainer(cfg, model)

    if not skip_preprocess:
        plan_and_preprocess(cfg, env, model)

    train(cfg, env, model, trainer, num_gpus=num_gpus, num_epochs=num_epochs)
