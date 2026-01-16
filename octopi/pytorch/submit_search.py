from __future__ import annotations

import torch, mlflow, optuna, os, time, traceback, pprint
from octopi.pytorch.search import ModelExplorer
from octopi.utils import parsers, io
from optuna.trial import TrialState
import torch.multiprocessing as mp
from octopi.datasets import config
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

# -----------------------------
# Supervisor / worker settings
# -----------------------------
POLL_S = 5
MAX_RESTARTS_PER_GPU = 25
CHECK_DB_EVERY_S = 120 # 2 minutes, to detect external study changes

@dataclass
class WorkerSpec:
    gpu_id: int
    proc: Optional[mp.Process] = None
    restarts: int = 0

# -----------------------------
# Optuna utilities
# -----------------------------

TERMINAL_STATES = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)

def count_terminal_trials(study) -> int:
    return len(study.get_trials(states=TERMINAL_STATES))

def make_storage(storage_url: str):
    # For multi-worker: prefer Postgres/MySQL. SQLite can lock.
    return optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
        engine_kwargs={
            "connect_args": {
                "timeout": 300,  # 5 minutes timeout for lock acquisition
                "check_same_thread": False  # Allow multi-threaded access
            },
            "pool_pre_ping": True  # Verify connections before using
        }
    )

def get_sampler():
    """Returns Optuna's TPE sampler with default settings."""
    return optuna.samplers.TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True
    )

def get_pruner(val_interval: int, n_warmup_steps: int = 300):
    """Returns Optuna's pruning strategy."""
    return optuna.pruners.MedianPruner(
        n_startup_trials=10,    # let at least 10 full trials run before pruning
        n_warmup_steps=n_warmup_steps,     # dont prune before (nepochs / 3) epochs
        interval_steps=val_interval       # check each interval
    )    

def get_study(study_name: str, storage: str, val_interval: int, n_warmup_steps: int = 300):
    """Returns the Optuna study object."""
    return optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=get_sampler(),
            pruner=get_pruner(val_interval, n_warmup_steps),
            load_if_exists=True,
        )

# -----------------------------
# Worker Process: one GPU, loop trials
# -----------------------------
def gpu_worker_loop(
    gpu_id: int,
    storage_url: str,
    study_name: str,
    submit_kwargs: dict,
    stop_event: mp.Event,
    ):
    """
    Runs forever (until stop_event set). Handles prunes/fails without exiting.
    If the *process* crashes, supervisor restarts it.
    """
    # Pin to GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # Log worker information
    print(f"[worker {gpu_id}] pid={os.getpid()} device={device}", flush=True)

    # Each worker creates its own storage/study handle to avoid SQLite locking issues
    storage = make_storage(storage_url)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Main loop: keep asking for new trials until enough trials have been completed or something breaks
    while not stop_event.is_set():
        try:
            # Ask for a new trial
            trial = study.ask()

            # Verbose to show data splits (# runs / tomograms) for only the first trial
            if trial.number == 0: verbose = True
            else: verbose = False

            # Build datamodule per-worker
            cfg = config.DataGeneratorConfig.from_dict(submit_kwargs)
            data_generator = cfg.create_data_generator(verbose=verbose)
            model_search = ModelExplorer(data_generator, submit_kwargs["model_type"], submit_kwargs["output"])            

            try:
                value = model_search.objective(
                    trial=trial,
                    epochs=int(submit_kwargs.get("num_epochs", 100)),
                    device=device,
                    val_interval=int(submit_kwargs.get("val_interval", 10)),
                    best_metric=str(submit_kwargs.get("best_metric", "avg_f1")),
                )
                study.tell(trial, value)
                print(f"[worker {gpu_id}] COMPLETE trial={trial.number} value={value}", flush=True)

            except optuna.TrialPruned:
                study.tell(trial, state=TrialState.PRUNED)
                print(f"[worker {gpu_id}] PRUNED trial={trial.number}", flush=True)

            except torch.cuda.OutOfMemoryError:
                study.tell(trial, state=TrialState.FAIL)
                print(f"[worker {gpu_id}] FAILED(OOM) trial={trial.number}", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)

            except Exception as e:
                study.tell(trial, state=TrialState.FAIL)
                print(f"[worker {gpu_id}] FAILED(EXC) trial={trial.number}: {e}", flush=True)
                traceback.print_exc()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)

        except Exception as e:
            # This catches Optuna/DB-level issues (e.g. sqlite lock)
            print(f"[worker {gpu_id}] LOOP ERROR: {e}", flush=True)
            traceback.print_exc()
            time.sleep(5)

# -------------------------
# Model Search Submitter
# -------------------------

class ExploreSubmitter:
    def __init__(
        self,
        copick_config: str,
        target_name: str, target_user_id: str, target_session_id: str,
        tomo_algorithm: str, voxel_size: float,
        model_type: str, best_metric: str = 'avg_f1',
        num_epochs: int = 1000, num_trials: int = 100,
        data_split: str = 0.8, random_seed: int = 42,
        val_interval: int = 10,
        ntomo_cache: int = 15,
        trainRunIDs: List[str] = None, validateRunIDs: List[str] = None,
        study_name: str = 'explore',
        background_ratio: float = 0.0
    ):
        """
        Initialize the ModelSearch class for architecture search with Optuna.

        Args:
            copick_config (str or dict): Path to the CoPick configuration file or a dictionary for multi-config training.
            target_name (str): Name of the target for segmentation.
            target_user_id (str): Optional user ID for tracking.
            target_session_id (str): Optional session ID for tracking.
            tomo_algorithm (str): Tomogram algorithm to use.
            voxel_size (float): Voxel size for tomograms.
            Nclass (int): Number of prediction classes.
            model_type (str): Type of model to use.
            random_seed (int): Seed for reproducibility.
            num_epochs (int): Number of epochs per trial.
            num_trials (int): Number of trials for hyperparameter optimization.
            ntomo_cache (int): Batch size for tomogram loading.
            best_metric (str): Metric to optimize.
            val_interval (int): Validation interval.
            trainRunIDs (List[str]): List of training run IDs.
            validateRunIDs (List[str]): List of validation run IDs.
            data_split (str): Data split ratios.
            background_ratio (float): Background ratio for data augmentation.
        """

        # Input parameters 
        self.copick_config = copick_config
        self.target_name = target_name
        self.target_user_id = target_user_id
        self.target_session_id = target_session_id
        self.tomo_algorithm = tomo_algorithm
        self.voxel_size = voxel_size
        self.model_type = model_type
        self.random_seed = random_seed
        self.num_epochs = num_epochs
        self.n_warmup_steps = int(num_epochs / 3)
        self.num_trials = num_trials
        self.ntomo_cache = ntomo_cache
        self.best_metric = best_metric
        self.val_interval = val_interval
        self.trainRunIDs = trainRunIDs
        self.validateRunIDs = validateRunIDs
        self.data_split = data_split
        self.background_ratio = background_ratio

    def run_model_search(self, study_name='octopi_nas', output='explore_results'):
        """Performs model architecture search using Optuna and MLflow.

        Parent process keeps one child per GPU alive.
        If a child exits unexpectedly, restart it.
        """

        # Set random seed for reproducibility
        config.set_seed(self.random_seed)

        # Get list of available GPU IDs
        gpu_ids = list(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError("No GPUs visible. If you expect GPUs, check CUDA setup.")

        # Get the parameters
        submit_kwargs = self.get_parameters(study_name)
        submit_kwargs["output"] = output
        self.save_parameters(submit_kwargs, output)

        # Create Output Directory and Optuna Study
        os.makedirs(output, exist_ok=True)
        storage_url = f"sqlite:///{output}/trials.db"
        storage = make_storage(storage_url)
        study = get_study(study_name, storage, self.val_interval, self.n_warmup_steps)

        # Set the MLflow experiment 
        mlflow.set_experiment(study_name)

        # Supervisor process: start one worker process per GPU
        mp.set_start_method("spawn", force=True)
        stop_event = mp.Event()
        specs = {gid: WorkerSpec(gpu_id=gid) for gid in gpu_ids}

        # Function to start a worker process
        def start_worker(gid: int):
            spec = specs[gid]
            p = mp.Process(
                target=gpu_worker_loop,
                args=(gid, storage_url, study_name, submit_kwargs, stop_event),
                daemon=False,
            )
            p.start()
            spec.proc = p
            print(f"[supervisor] started worker gpu={gid} pid={p.pid}", flush=True)

        # Start all GPU workers
        for gid in gpu_ids:
            start_worker(gid)

        # Monitor the workers and manage the study
        last_db_check = 0.0
        try:
            while True:

                # --- cheap: monitor + restart dead workers ---
                for gid, spec in specs.items():
                    p = spec.proc
                    if p is None:
                        continue
                    if not p.is_alive():
                        code = p.exitcode
                        print(f"[supervisor] worker gpu={gid} died exitcode={code}", flush=True)
                        spec.restarts += 1
                        if spec.restarts > MAX_RESTARTS_PER_GPU:
                            raise RuntimeError(f"worker on gpu {gid} exceeded restart limit ({MAX_RESTARTS_PER_GPU})")
                        start_worker(gid)

                # --- expensive: touch DB occasionally ---
                now = time.time()
                if now - last_db_check >= CHECK_DB_EVERY_S:
                    last_db_check = now
                    try:
                        # reload study view (optional but good)
                        study = optuna.load_study(study_name=study_name, storage=storage)
                        done = count_terminal_trials(study)
                        print(f"[supervisor] done={done}/{self.num_trials}", flush=True)

                        if done >= self.num_trials:
                            print(f"[supervisor] reached {done} trials >= target {self.num_trials}. stopping.", flush=True)
                            break

                    except Exception as e:
                        # handle db locked, transient net fs weirdness, etc.
                        if "database is locked" in str(e).lower():
                            print("[supervisor] DB locked; will retry later.", flush=True)
                        else:
                            raise

                # Sleep for a short interval before checking again
                time.sleep(POLL_S)
        except KeyboardInterrupt:
            print("[supervisor] Ctrl-C received. stopping.", flush=True)
        finally:
            stop_event.set()
            # Give workers a moment to exit gracefully
            time.sleep(2)
            for gid, spec in specs.items():
                if spec.proc and spec.proc.is_alive():
                    print(f"[supervisor] terminating worker gpu={gid} pid={spec.proc.pid}", flush=True)
                    spec.proc.terminate()
            for gid, spec in specs.items():
                if spec.proc:
                    spec.proc.join(timeout=30)
            print("[supervisor] done.", flush=True)

        # Save the contour plot and results
        self.save_contour_plot_as_png(study, output)

    def save_contour_plot_as_png(self, study, output):
        """
        Save the contour plot of hyperparameter interactions as a PNG, 
        automatically extracting parameter names from the study object.

        Args:
            study: The Optuna study object.
        """
        # Extract all parameter names from the study trials
        all_params = set()
        for trial in study.trials:
            all_params.update(trial.params.keys())
        all_params = list(all_params)  # Convert to a sorted list for consistency

        # Generate the contour plot
        fig = optuna.visualization.plot_contour(study, params=all_params) 

        # Adjust figure size and font size
        fig.update_layout(
            width=6000, height=6000,  # Large figure size
            font=dict(size=40)  # Increase font size for better readability
        )

        # Save the plot as a PNG file
        fig.write_image(f'{output}/contour_plot.png', scale=1)  

        # Extract trial data
        trials = [
            {**trial.params, 'objective_value': trial.value}
            for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        # Convert to DataFrame
        df = pd.DataFrame(trials)

        # Save to CSV
        df.to_csv(f"{output}/optuna_results.csv", index=False)

    def get_parameters(self, study_name: str) -> dict:
        """Returns the parameters of the model search submitter."""
        return dict(
            config=self.copick_config,
            name=self.target_name,user_id=self.target_user_id, session_id=self.target_session_id,
            tomo_algorithm=self.tomo_algorithm, voxel_size=self.voxel_size,
            model_type=self.model_type, best_metric=self.best_metric, num_epochs=self.num_epochs,
            num_trials=self.num_trials, background_ratio=self.background_ratio,
            data_split=self.data_split, random_seed=self.random_seed,
            val_interval=self.val_interval, ntomo_cache=self.ntomo_cache,
            trainRunIDs=self.trainRunIDs, validateRunIDs=self.validateRunIDs,
            study_name=study_name
        )

    def save_parameters(self, params, output):
        """
        Display the Parameters and Save to Output Path
        """

        target_info = [self.target_name, self.target_user_id, self.target_session_id]
        output_params = {
            'input': {
                'config': self.copick_config, 'target_info': target_info, 
                'tomo_algorithm': self.tomo_algorithm, 'voxel_size': self.voxel_size },
            'optimization': {
                'model_type': self.model_type, 'random_seed': self.random_seed, 
                'num_trials': self.num_trials, 'best_metric': self.best_metric },
            'training': {
                'num_epochs': self.num_epochs, 'ntomo_cache': self.ntomo_cache, 
                'trainRunIDs': self.trainRunIDs, 'validateRunIDs': self.validateRunIDs, 
                'data_split': self.data_split }
        }

        # Print the Parameters
        print('Parameters for Model Architecture Search: ')
        pprint.pprint(output_params); print()

        # Save to YAML File
        io.save_parameters_yaml(output_params, f'{output}/model-search.yaml')