"""
nnUNet inference on CoPick tomograms.

Usage
-----
Single tomogram:
    from octopi.nnunet.predict import nnUNetPredictor
    model = nnUNetPredictor(plans="plans.json", dataset_json="dataset.json", weights="checkpoint_best.pth")
    seg   = model.predict(tomogram)          # np.ndarray (Z, Y, X) uint8

Full CoPick project (auto multi-GPU):
    model.batch_predict(copick_config="config.json", tomogram_uri="wbp@10.0", seg_name="nnunet")

CLI:
    octopi nnunet segment -c config.json -p plans.json -d dataset.json -w checkpoint.pth -uri wbp@10.0 -n nnunet
"""
import os, warnings

# nnunetv2 emits warnings when its path env-vars are unset; irrelevant for inference.
warnings.filterwarnings("ignore", message="nnUNet_raw")
warnings.filterwarnings("ignore", message="nnUNet_preprocessed")
warnings.filterwarnings("ignore", message="nnUNet_results")

# Silence NumExpr thread-count notices.
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

from octopi.nnunet.train import MODEL_TO_TRAINER
import rich_click as click

# ── trainer class resolution ─────────────────────────────────────────────────

def _resolve_trainer_class(trainer_name: str):
    """
    Return the trainer class for the given name.

    For standard nnUNet trainers we do a direct import (fast).
    For MedNeXt trainers we fall back to recursive_find_python_class, which
    scans the trainer directory — slow (~30-60 s) but only needed once per
    session and only when a MedNeXt checkpoint is used.
    """
    # Fast path: standard nnUNet trainers — format is "dotted.module.path::ClassName"
    # The module path points to the .py file; the class name is after "::".
    _DIRECT = {
        "nnUNetTrainer":
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer::nnUNetTrainer",
        "nnUNetTrainerNoMirroring":
            "nnunetv2.training.nnUNetTrainer.variants.training_length_and_nsteps.nnUNetTrainerNoMirroring::nnUNetTrainerNoMirroring",
    }
    if trainer_name in _DIRECT:
        import importlib
        module_path, class_name = _DIRECT[trainer_name].split("::")
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

class single_gpu_nnUNetPredictor:
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
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor as _Pred
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        import inspect, json, torch
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
        return seg.astype(np.uint8)

    def batch_predict(
        self,
        copick_config: str,
        tomogram_uri: str,
        seg_name: str,
        user_id: str = "nnunet",
        session_id: str = "1",
        run_ids=None,
    ):
        """
        Run inference on CoPick runs and write predictions back as segmentations.

        Parameters
        ----------
        tomogram_uri : str
            URI in ``algorithm@voxel_size`` format, e.g. ``"wbp@10.0"``.
        run_ids : list[str] | None
            Specific run names to process.  None = all runs in the project.
        """
        import copick
        from copick_utils.io import writers
        from copick.util.uri import resolve_copick_objects
        from tqdm import tqdm

        tomo_algorithm, voxel_size_str = tomogram_uri.split("@")
        voxel_size = float(voxel_size_str)

        root  = copick.from_file(copick_config)
        runs  = root.runs if run_ids is None else [root.get_run(r) for r in run_ids]

        for run in tqdm(runs, desc="Running nnUNet inference"):
            if run is None:
                continue
            vols = resolve_copick_objects(tomogram_uri, root, "tomogram", run_name=run.name)
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


# ── multi-GPU inference ───────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class _NnUNetJobSpec:
    plans: str
    dataset_json: str
    weights: list
    tile_step_size: float
    use_mirroring: bool
    copick_config: str
    tomogram_uri: str
    seg_name: str
    user_id: str
    session_id: str
    run_ids: list = field(default_factory=list)


def _nnunet_worker(rank: int, jobs: list):
    """One process per GPU — loads the model on cuda:{rank} and processes its shard."""
    import torch
    torch.cuda.set_device(rank)
    import os
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count())))

    job = jobs[rank]
    predictor = single_gpu_nnUNetPredictor(
        plans=job.plans,
        dataset_json=job.dataset_json,
        weights=job.weights,
        tile_step_size=job.tile_step_size,
        use_mirroring=job.use_mirroring,
        device=torch.device(f"cuda:{rank}"),
    )
    if job.run_ids:
        predictor.batch_predict(
            copick_config=job.copick_config,
            tomogram_uri=job.tomogram_uri,
            seg_name=job.seg_name,
            user_id=job.user_id,
            session_id=job.session_id,
            run_ids=job.run_ids,
        )


class nnUNetPredictor:
    """
    Universal nnUNet predictor — automatically uses all available GPUs for batch_predict.

    For single-tomogram inference (predict) the model is loaded on-demand on cuda:0.
    For batch inference (batch_predict) run IDs are sharded round-robin across all GPUs
    using mp.spawn (one process per GPU, CUDA-safe).

    Usage
    -----
    predictor = nnUNetPredictor(plans=..., dataset_json=..., weights=...)
    seg = predictor.predict(tomogram)                            # single volume, single GPU
    predictor.batch_predict(copick_config=..., tomogram_uri=..., seg_name=...)  # all GPUs
    """

    def __init__(
        self,
        plans: str,
        dataset_json: str,
        weights: "str | list[str]",
        tile_step_size: float = 0.5,
        use_mirroring: bool = True,
    ):
        self.plans          = plans
        self.dataset_json   = dataset_json
        self.weights        = [weights] if isinstance(weights, str) else weights
        self.tile_step_size = tile_step_size
        self.use_mirroring  = use_mirroring
        self._single        = None   # lazy-loaded on first predict() call

    def _get_single_predictor(self):
        if self._single is None:
            self._single = single_gpu_nnUNetPredictor(
                plans=self.plans,
                dataset_json=self.dataset_json,
                weights=self.weights,
                tile_step_size=self.tile_step_size,
                use_mirroring=self.use_mirroring,
            )
        return self._single

    def predict(self, tomogram: "np.ndarray", voxel_size_angstrom: float = 10.0) -> "np.ndarray":
        """Predict segmentation for a single tomogram on cuda:0."""
        return self._get_single_predictor().predict(tomogram, voxel_size_angstrom)

    def batch_predict(
        self,
        copick_config: str,
        tomogram_uri: str,
        seg_name: str,
        user_id: str = "nnunet",
        session_id: str = "1",
        run_ids=None,
    ):
        """
        Run inference on CoPick runs across all available GPUs.

        Parameters
        ----------
        tomogram_uri : str
            URI in ``algorithm@voxel_size`` format, e.g. ``"wbp@10.0"``.
        run_ids : list[str] | None
            Specific run names to process.  None = all runs in the project.
        """
        import torch, copick
        import torch.multiprocessing as mp

        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA GPUs available.")

        if run_ids is None:
            root    = copick.from_file(copick_config)
            run_ids = [r.name for r in root.runs]

        # Single-GPU: skip subprocess overhead, use lazy single predictor
        if world_size == 1:
            self._get_single_predictor().batch_predict(
                copick_config=copick_config,
                tomogram_uri=tomogram_uri,
                seg_name=seg_name,
                user_id=user_id,
                session_id=session_id,
                run_ids=run_ids,
            )
            return

        # Multi-GPU: shard run IDs round-robin, one process per GPU
        shards = [[] for _ in range(world_size)]
        for i, rid in enumerate(run_ids):
            shards[i % world_size].append(rid)

        jobs = [
            _NnUNetJobSpec(
                plans=self.plans,
                dataset_json=self.dataset_json,
                weights=self.weights,
                tile_step_size=self.tile_step_size,
                use_mirroring=self.use_mirroring,
                copick_config=copick_config,
                tomogram_uri=tomogram_uri,
                seg_name=seg_name,
                user_id=user_id,
                session_id=session_id,
                run_ids=shard,
            )
            for shard in shards
        ]

        mp.spawn(_nnunet_worker, args=(jobs,), nprocs=world_size, join=True)
        print("Done writing predictions to CoPick.")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command("segment", no_args_is_help=True)
# Input Arguments
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Path to copick config.json")
@click.option('-p', '--plans', required=True, type=click.Path(exists=True), help="Path to nnunet plans.json")
@click.option('-d', '--dataset', required=True, type=click.Path(exists=True), help="Path to nnunet dataset.json")
@click.option("-w","--weights", required=True, type=click.Path(exists=True), help="Path to nnunet checkpoint.pth")
@click.option('-uri', "--uri", type=str, default="wbp@10.0", help="Tomogram URI to predict")
@click.option("--tta", type=bool, default=True, help="Enable mirroring TTA.")
@click.option("--run-ids", type=str, default=None,
              help="CoPick run IDs to predict (as comma-separated list).")
# Output Arguments
@click.option('-n', '--name', type=str, required=True, help="Name of Segmentation")
@click.option('-uid', '--user-id', type=str, default="nnunet", help="User ID")
@click.option('-sid', '--session-id', type=str, default="1", help="Session ID")
def cli(config, plans, dataset, uri, weights, tta, run_ids, name, user_id, session_id):
    """Run nnUNet inference on CoPick tomograms and write predictions back."""
    
    run_predict(config, plans, dataset, uri,  weights, tta, run_ids, name, user_id, session_id)
    
def run_predict(config, plans, dataset, uri, weights, tta, run_ids, name, user_id, session_id):
    
    run_ids_list = run_ids.split(',') if run_ids else None   

    predictor = nnUNetPredictor(
        plans=plans,
        dataset_json=dataset,
        weights=weights,
        use_mirroring=tta,
    )

    predictor.batch_predict(
        copick_config=config,
        tomogram_uri=uri,
        seg_name=name,
        user_id=user_id,
        session_id=session_id,
        run_ids=run_ids_list,
    )
