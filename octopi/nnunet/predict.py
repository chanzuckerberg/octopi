"""
nnUNet inference on CoPick tomograms.

Usage
-----
Single tomogram:
    from octopi.nnunet.predict import NnUNetPredictor
    model = NnUNetPredictor(plans="plans.json", dataset_json="dataset.json", weights="checkpoint_best.pth")
    seg   = model.predict(tomogram)          # np.ndarray (Z, Y, X) uint8

Full CoPick project:
    model.predict_copick(copick_config, tomo_algorithm, voxel_size, seg_name)

CLI:
    octopi nnunet predict -c config.yaml [--folds 0 --folds 1] [--run-ids TS_001]
"""
from octopi.nnunet.train import MODEL_TO_TRAINER, resolve_trainer
import rich_click as click


# ── helpers ──────────────────────────────────────────────────────────────────

def _model_folder(cfg: dict, trainer: str, model: str) -> str:
    """Resolve the nnUNet results folder for this trainer/model/configuration."""
    from pathlib import Path

    plans       = "nnUNetResEncUNetLPlans" if model == "resnecl" else "nnUNetPlans"
    config      = cfg.get("configuration", "3d_fullres")
    dataset_dir = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    folder = (
        Path(cfg["nnunet_results"])
        / dataset_dir
        / f"{trainer}__{plans}__{config}"
    )
    if not folder.exists():
        raise FileNotFoundError(
            f"Model folder not found: {folder}\n"
            "Run `octopi nnunet train` first."
        )
    return str(folder)


def _checkpoint_paths(model_folder: str, folds: tuple, checkpoint_name: str) -> list[str]:
    """Return the list of checkpoint paths for the requested folds."""
    from pathlib import Path
    paths = []
    for f in folds:
        p = Path(model_folder) / f"fold_{f}" / checkpoint_name
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        paths.append(str(p))
    return paths


# ── trainer class resolution ─────────────────────────────────────────────────

def _resolve_trainer_class(trainer_name: str):
    """
    Return the trainer class for the given name.

    For standard nnUNet trainers we do a direct import (fast).
    For MedNeXt trainers we fall back to recursive_find_python_class, which
    scans the trainer directory — slow (~30-60 s) but only needed once per
    session and only when a MedNeXt checkpoint is used.
    """
    # Fast path: standard nnUNet trainers
    _DIRECT = {
        "nnUNetTrainer": "nnunetv2.training.nnUNetTrainer.nnUNetTrainer",
        "nnUNetTrainerNoMirroring": "nnunetv2.training.nnUNetTrainer.variants.training_length_and_nsteps.nnUNetTrainerNoMirroring",
    }
    if trainer_name in _DIRECT:
        module_path, _, class_name = _DIRECT[trainer_name].rpartition(".")
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)

    # MedNeXt trainers live in the file we copy into nnunetv2's trainer dir
    if "MedNeXt" in trainer_name:
        try:
            from nnunetv2.training.nnUNetTrainer.variants import nnUNetTrainerMedNeXt as _mn_mod
            return getattr(_mn_mod, trainer_name)
        except (ImportError, AttributeError):
            pass

    # Slow fallback: walk every .py in the trainer tree (original nnUNet approach)
    import nnunetv2, os
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    cls = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if cls is None:
        raise RuntimeError(
            f"Trainer class '{trainer_name}' not found. "
            "For MedNeXt models, run `octopi nnunet train` once to register the trainer."
        )
    return cls


# ── predictor ────────────────────────────────────────────────────────────────

class nnUNetPredictor:
    """
    nnUNet inference wrapper that accepts individual files rather than a folder.

    Parameters
    ----------
    plans : str
        Path to plans.json (written by nnUNetv2_plan_and_preprocess).
    dataset_json : str
        Path to dataset.json.
    weights : str | list[str]
        Path(s) to checkpoint .pth file(s).  Multiple paths are ensembled
        by averaging logits before argmax — same as nnUNet's fold ensembling.
    tile_step_size : float
        Sliding-window step as a fraction of patch size (lower = more overlap).
    use_mirroring : bool
        Enable nnUNet's built-in mirroring TTA.
    device : torch.device | None
        Inference device.  Defaults to cuda:0 if available.
    """

    def __init__(
        self,
        plans: str,
        dataset_json: str,
        weights: "str | list[str]",
        tile_step_size: float = 0.5,
        use_mirroring: bool = True,
        device=None,
    ):
        import inspect, json, torch
        import nnunetv2
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor as _Pred
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        import os

        if device is None:
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        # ── load plans + dataset ──────────────────────────────────────────────
        with open(plans) as f:
            plans_dict = json.load(f)
        with open(dataset_json) as f:
            dataset_dict = json.load(f)

        plans_manager = PlansManager(plans_dict)

        # ── load checkpoint(s) ───────────────────────────────────────────────
        if isinstance(weights, str):
            weights = [weights]

        parameters = []
        trainer_name = configuration_name = mirroring_axes = None

        for i, w in enumerate(weights):
            ckpt = torch.load(w, map_location="cpu", weights_only=False)
            if i == 0:
                trainer_name       = ckpt["trainer_name"]
                configuration_name = ckpt["init_args"]["configuration"]
                mirroring_axes     = ckpt.get("inference_allowed_mirroring_axes")
            parameters.append(ckpt["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)

        # ── build network ────────────────────────────────────────────────────
        trainer_class = _resolve_trainer_class(trainer_name)

        num_input_channels  = determine_num_input_channels(plans_manager, configuration_manager, dataset_dict)
        num_output_channels = plans_manager.get_label_manager(dataset_dict).num_segmentation_heads

        # nnUNet v2 changed the build_network_architecture signature; handle both
        sig = inspect.signature(trainer_class.build_network_architecture)
        if "plans_manager" in sig.parameters:
            network = trainer_class.build_network_architecture(
                plans_manager, configuration_manager,
                num_input_channels, num_output_channels,
                enable_deep_supervision=False,
            )
        else:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels, num_output_channels,
                enable_deep_supervision=False,
            )

        # ── wire everything into the nnUNet predictor ─────────────────────────
        self._predictor = _Pred(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
        )
        self._predictor.manual_initialization(
            network=network,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            parameters=parameters,
            dataset_json=dataset_dict,
            trainer_name=trainer_name,
            inference_allowed_mirroring_axes=mirroring_axes,
        )

        print(f"Loaded nnUNet model  trainer={trainer_name}  config={configuration_name}  folds={len(weights)}  device={device}")

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, tomogram: "np.ndarray", voxel_size_angstrom: float = 10.0) -> "np.ndarray":
        """
        Predict segmentation for a single tomogram.

        Parameters
        ----------
        tomogram : np.ndarray, shape (Z, Y, X)
        voxel_size_angstrom : float
            Voxel spacing in Angstrom — converted to nm (÷10) for nnUNet.

        Returns
        -------
        np.ndarray, shape (Z, Y, X), dtype uint8
        """
        import numpy as np

        if tomogram.ndim == 3:
            tomogram = tomogram[np.newaxis]           # → (1, Z, Y, X)
        tomogram = tomogram.astype(np.float32)

        spacing_nm = float(voxel_size_angstrom) / 10.0
        props = {"spacing": [spacing_nm, spacing_nm, spacing_nm]}

        seg = self._predictor.predict_single_npy_array(
            tomogram, props,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        return seg.astype(np.uint16)

    def batch_predict(
        self,
        copick_config: str,
        tomo_algorithm: str,
        voxel_size: float,
        seg_name: str,
        user_id: str = "nnunet",
        session_id: str = "0",
        run_ids=None,
    ):
        """
        Run inference on CoPick runs and write predictions back as segmentations.

        Parameters
        ----------
        run_ids : list[str] | None
            Specific run names to process.  None = all runs in the project.
        """
        import copick, numpy as np
        from copick_utils.io import writers
        from copick.util.uri import resolve_copick_objects
        from tqdm import tqdm

        root    = copick.from_file(copick_config)
        vol_uri = f"{tomo_algorithm}@{voxel_size}"
        runs    = root.runs if run_ids is None else [root.get_run(r) for r in run_ids]

        for run in tqdm(runs, desc="Running nnUNet inference"):
            if run is None:
                continue
            vols = resolve_copick_objects(vol_uri, root, "tomogram", run_name=run.name)
            if not vols:
                print(f"  [SKIP] No tomogram found for run '{run.name}'")
                continue

            seg = self.predict(vols[0].numpy(), voxel_size_angstrom=voxel_size)
            writers.segmentation(
                run, seg,
                voxel_size=voxel_size,
                name=seg_name,
                user_id=user_id,
                session_id=session_id,
            )

        print("Done writing predictions to CoPick.")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command("predict", no_args_is_help=True)
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Path to nnunet config.yaml")
@click.option("--model", type=click.Choice(list(MODEL_TO_TRAINER)), default=None,
              help="Model — must match training.")
@click.option("--folds", multiple=True, type=int, default=[0], show_default=True,
              help="Fold(s) to ensemble (repeat flag: --folds 0 --folds 1).")
@click.option("--checkpoint", default="checkpoint_best.pth", show_default=True,
              help="Checkpoint filename inside each fold directory.")
@click.option("--tile-step-size", default=0.5, show_default=True, type=float,
              help="Tile step size as fraction of patch size.")
@click.option("--no-tta", is_flag=True, default=False, help="Disable mirroring TTA.")
@click.option("--run-ids", multiple=True, default=None,
              help="CoPick run IDs to predict (default: all runs).")
def cli(config, model, folds, checkpoint, tile_step_size, no_tta, run_ids):
    """Run nnUNet inference on CoPick tomograms and write predictions back."""
    from octopi.nnunet.utils import _load_config

    cfg            = _load_config(config)
    model, trainer = resolve_trainer(cfg, model)
    folder         = _model_folder(cfg, trainer, model)
    weight_paths   = _checkpoint_paths(folder, tuple(folds) if folds else (0,), checkpoint)

    plans_path      = f"{folder}/plans.json"
    dataset_path    = f"{folder}/dataset.json"

    predictor = nnUNetPredictor(
        plans=plans_path,
        dataset_json=dataset_path,
        weights=weight_paths,
        tile_step_size=tile_step_size,
        use_mirroring=not no_tta,
    )

    predictor.predict_copick(
        copick_config=cfg["copick_config"],
        tomo_algorithm=cfg["tomo_algorithm"],
        voxel_size=cfg["voxel_size"],
        seg_name=cfg.get("segmentation_name", "nnunet"),
        user_id=cfg.get("prediction_user_id", "nnunet"),
        session_id=cfg.get("prediction_session_id", "0"),
        run_ids=list(run_ids) if run_ids else None,
    )
