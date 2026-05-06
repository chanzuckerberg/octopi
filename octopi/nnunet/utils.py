import subprocess, sys, yaml


MEDNEXT_INSTALL = "pip install git+https://github.com/MIC-DKFZ/MedNeXt.git"

def check_mednext_installed():
    try:
        import nnunet_mednext  # noqa: F401
    except ModuleNotFoundError:
        print("[ERROR] MedNeXt is not installed. Run:")
        print(f"  {MEDNEXT_INSTALL}")
        sys.exit(1)
    _register_mednext_trainer()

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _run(cmd: list[str], env: dict):

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def _register_mednext_trainer():
    """Copy octopi's MedNeXt trainer into nnunetv2's trainer discovery directory."""
    from pathlib import Path
    import shutil, nnunetv2

    src = Path(__file__).parent / "mednext_trainer.py"
    dst = (Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
           / "variants" / "nnUNetTrainerMedNeXt.py")

    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  [INFO] Registered MedNeXt trainer into nnUNet v2.")

def _scale_batch_size_for_ddp(cfg: dict, model: str, num_gpus: int):
    """
    Scale the plans JSON batch_size to num_gpus × planned_per_gpu_batch_size.

    nnUNet plans for single-GPU memory: batch_size=2 means 2 samples per GPU.
    In DDP the global batch is split across GPUs, so we multiply by num_gpus
    so each GPU still processes the same number of samples (same memory footprint).
    The updated plans file is written back in-place; re-run plan_and_preprocess to reset.
    """
    from pathlib import Path
    import json

    plans_name    = "nnUNetResEncUNetLPlans" if model == "resnecl" else "nnUNetPlans"
    configuration = cfg.get("configuration", "3d_fullres")
    dataset_dir   = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    plans_file    = Path(cfg["nnunet_preprocessed"]) / dataset_dir / f"{plans_name}.json"

    if not plans_file.exists():
        return  # preprocessing hasn't run yet; nnUNet will catch this itself

    with open(plans_file) as f:
        plans = json.load(f)

    try:
        current = plans["configurations"][configuration]["batch_size"]
    except KeyError:
        return

    # On first scale we store the planner's per-GPU value so that resume runs
    # always derive scaled from the original single-GPU batch size, not from a
    # previously-scaled value (which would compound on every resume).
    per_gpu_key = "_octopi_per_gpu_batch_size"
    per_gpu_bs  = plans.get(per_gpu_key, current)
    scaled      = per_gpu_bs * num_gpus

    if current == scaled:
        return  # already correct

    plans[per_gpu_key] = per_gpu_bs
    plans["configurations"][configuration]["batch_size"] = scaled
    with open(plans_file, "w") as f:
        json.dump(plans, f, indent=4)

    print(
        f"  [plans] batch_size scaled {per_gpu_bs} → {scaled} "
        f"({per_gpu_bs}/GPU × {num_gpus} GPUs, same per-GPU memory)."
    )
