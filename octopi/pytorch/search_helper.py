from __future__ import annotations

import torch, mlflow, optuna, os, time, traceback
from octopi.pytorch.search import ModelExplorer
from optuna.trial import TrialState
import torch.multiprocessing as mp
from octopi.datasets import config
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# Supervisor / worker settings
# -----------------------------
POLL_S = 5
MAX_RESTARTS_PER_GPU = 25
CHECK_DB_EVERY_S = 120 # 2 minutes, to detect external study changes
PRINT_EVERY_S = 60 # print submitit once per minute

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


# -----------------------------
# One-trial job (for submitit; must be top-level for pickling)
# -----------------------------
def run_one_trial(storage_url: str, study_name: str, submit_kwargs: dict):
    """
    Run a single Optuna trial. Used as the submitit job target.
    Connects to the study, asks for a trial, runs ModelExplorer.objective, tells the result.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    print(f"[run_one_trial] pid={os.getpid()} device={device}", flush=True)

    storage = make_storage(storage_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = study.ask()
    print(f"[run_one_trial] START trial={trial.number}", flush=True)

    verbose = trial.number == 0
    cfg = config.DataGeneratorConfig.from_dict(submit_kwargs)
    data_generator = cfg.create_data_generator(verbose=verbose)
    model_search = ModelExplorer(
        data_generator, submit_kwargs["model_type"], submit_kwargs["output"]
    )

    try:
        value = model_search.objective(
            trial=trial,
            epochs=int(submit_kwargs.get("num_epochs", 100)),
            device=device,
            val_interval=int(submit_kwargs.get("val_interval", 10)),
            best_metric=str(submit_kwargs.get("best_metric", "avg_f1")),
        )
        study.tell(trial, value)
        print(f"[run_one_trial] COMPLETE trial={trial.number} value={value}", flush=True)
        return value
    except optuna.TrialPruned:
        study.tell(trial, state=TrialState.PRUNED)
        print(f"[run_one_trial] PRUNED trial={trial.number}", flush=True)
        raise
    except torch.cuda.OutOfMemoryError:
        study.tell(trial, state=TrialState.FAIL)
        print(f"[run_one_trial] FAILED(OOM) trial={trial.number}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise
    except Exception as e:
        study.tell(trial, state=TrialState.FAIL)
        print(f"[run_one_trial] FAILED(EXC) trial={trial.number}: {e}", flush=True)
        traceback.print_exc()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise