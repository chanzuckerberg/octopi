from __future__ import annotations

import torch, mlflow, optuna, os, time, pprint, submitit
import octopi.pytorch.search_helper as helper
import torch.multiprocessing as mp
from octopi.datasets import config
from octopi.utils import io
from typing import List
import pandas as pd

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

    def _setup_study_and_storage(self, study_name: str, output: str):
        """Create output dir, Optuna study, and save parameters. Returns (storage_url, storage, study, submit_kwargs)."""
        config.set_seed(self.random_seed)
        submit_kwargs = self.get_parameters(study_name)
        submit_kwargs["output"] = output
        self.save_parameters(submit_kwargs, output)
        os.makedirs(output, exist_ok=True)
        storage_url = f"sqlite:///{output}/trials.db"
        storage = helper.make_storage(storage_url)
        study = helper.get_study(study_name, storage, self.val_interval, self.n_warmup_steps)
        mlflow.set_experiment(study_name)
        return storage_url, storage, study, submit_kwargs

    def _run_worker_pool(
        self,
        storage_url: str,
        study_name: str,
        submit_kwargs: dict,
    ):
        """Run the worker pool until num_trials are done. Override in subclasses (e.g. submitit)."""
        gpu_ids = list(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError("No GPUs visible. If you expect GPUs, check CUDA setup.")

        mp.set_start_method("spawn", force=True)
        stop_event = mp.Event()
        specs = {gid: helper.WorkerSpec(gpu_id=gid) for gid in gpu_ids}

        def start_worker(gid: int):
            spec = specs[gid]
            p = mp.Process(
                target=helper.gpu_worker_loop,
                args=(gid, storage_url, study_name, submit_kwargs, stop_event),
                daemon=False,
            )
            p.start()
            spec.proc = p
            print(f"[supervisor] started worker gpu={gid} pid={p.pid}", flush=True)

        for gid in gpu_ids:
            start_worker(gid)

        last_db_check = 0.0
        storage = helper.make_storage(storage_url)
        try:
            while True:
                for gid, spec in specs.items():
                    p = spec.proc
                    if p is None:
                        continue
                    if not p.is_alive():
                        code = p.exitcode
                        print(f"[supervisor] worker gpu={gid} died exitcode={code}", flush=True)
                        spec.restarts += 1
                        if spec.restarts > helper.MAX_RESTARTS_PER_GPU:
                            raise RuntimeError(
                                f"worker on gpu {gid} exceeded restart limit ({helper.MAX_RESTARTS_PER_GPU})"
                            )
                        start_worker(gid)

                now = time.time()
                if now - last_db_check >= helper.CHECK_DB_EVERY_S:
                    last_db_check = now
                    try:
                        study = optuna.load_study(study_name=study_name, storage=storage)
                        done = helper.count_terminal_trials(study)
                        print(f"[supervisor] done={done}/{self.num_trials}", flush=True)
                        if done >= self.num_trials:
                            print(
                                f"[supervisor] reached {done} trials >= target {self.num_trials}. stopping.",
                                flush=True,
                            )
                            break
                    except Exception as e:
                        if "database is locked" in str(e).lower():
                            print("[supervisor] DB locked; will retry later.", flush=True)
                        else:
                            raise

                time.sleep(helper.POLL_S)
        except KeyboardInterrupt:
            print("[supervisor] Ctrl-C received. stopping.", flush=True)
        finally:
            stop_event.set()
            time.sleep(2)
            for gid, spec in specs.items():
                if spec.proc and spec.proc.is_alive():
                    print(
                        f"[supervisor] terminating worker gpu={gid} pid={spec.proc.pid}",
                        flush=True,
                    )
                    spec.proc.terminate()
            for gid, spec in specs.items():
                if spec.proc:
                    spec.proc.join(timeout=30)
            print("[supervisor] done.", flush=True)

    def run_model_search(self, study_name: str = 'octopi_nas', output: str = 'explore_results'):
        """Performs model architecture search using Optuna and MLflow.

        Sets up study/storage, runs worker pool (local or submitit via override), then saves contour plot.
        """
        storage_url, storage, _study, submit_kwargs = self._setup_study_and_storage(
            study_name, output
        )
        self._run_worker_pool(storage_url, study_name, submit_kwargs)
        study = optuna.load_study(study_name=study_name, storage=storage)
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


# -------------------------
# Submitit-backed Model Search (SLURM jobs; refill pool as jobs complete)
# -------------------------

class SubmititExplorer(ExploreSubmitter):
    """Model architecture search via submitit: submit N concurrent SLURM jobs, refill as they complete."""
    def __init__(
        self,
        n_concurrent_jobs: int = 5,
        cpus_per_task: int = 4,
        mem_per_cpu: int = 16,
        slurm_timeout_min: int = 1080,
        gpu_constraint: str = None,
        submitit_folder: str = "submitit_logs",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_concurrent_jobs = n_concurrent_jobs
        self.slurm_timeout_min = slurm_timeout_min
        self.slurm_cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu
        self.submitit_folder = submitit_folder
        self.gpu_constraint = gpu_constraint

        # Check to Make sure GPU Constraint is Valid
        gpu_constraint = helper.check_gpus(gpu_constraint)

        print('🚀 Using Submitit Explorer for Model Architecture Search with the following settings:')
        print(f"  - Concurrent Jobs: {self.n_concurrent_jobs}")
        print(f"  - Compute Constraint (cpus, mem_per_cpu_gb): {self.slurm_cpus_per_task}, {self.mem_per_cpu_gb}")
        print(f"  - SLURM Timeout (min): {self.slurm_timeout_min}")
        print(f"  - Submitit Folder: {self.submitit_folder}")
        print(f"  - GPU Constraint: {self.gpu_constraint}\n")

    def _run_worker_pool(
        self,
        storage_url: str,
        study_name: str,
        submit_kwargs: dict,
    ):
        """Run the submitit job pool until num_trials are done."""
        log_dir = os.path.join(submit_kwargs["output"], self.submitit_folder)
        os.makedirs(log_dir, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(
            slurm_partition="gpu",
            slurm_use_srun = False,
            slurm_job_name=study_name,
            timeout_min=self.slurm_timeout_min,
            cpus_per_task=self.slurm_cpus_per_task,
            slurm_mem_per_cpu=f"{self.mem_per_cpu_gb}G",
            slurm_additional_parameters={
                "gpus": "1",
            },
        )
        if self.gpu_constraint: # Optional GPU Constraint
            executor.update_parameters(
                slurm_constraint=f"{self.gpu_constraint}"
            )

        # Printing status every helper.PRINT_EVERY_S seconds
        last_print = 0

        # Start the submitit job pool
        storage = helper.make_storage(storage_url)
        running = []
        num_trials = self.num_trials

        def submit_one():
            return executor.submit(helper.run_one_trial, storage_url, study_name, submit_kwargs)

        # Submit initial batch (up to n_concurrent_jobs)
        for _ in range(min(self.n_concurrent_jobs, num_trials)):
            job = submit_one()
            running.append(job)
            print(f"[submitit] submitted job {job.job_id}", flush=True)

        # Refill as jobs complete until we have enough trials
        while True:
            done = [j for j in running if j.done()]
            for j in done:
                try:
                    j.result()
                except Exception as e:
                    print(f"[submitit] job {j.job_id} finished with error: {e}", flush=True)
                running.remove(j)

            # Check study progress
            study = optuna.load_study(study_name=study_name, storage=storage)
            done_count = helper.count_terminal_trials(study)

            # Print status periodically
            now = time.time()
            if now - last_print >= helper.PRINT_EVERY_S:
                print(
                    f"[submitit] done={done_count}/{num_trials} running={len(running)}",
                    flush=True,
                )
                last_print = now

            if done_count >= num_trials and not running:
                break

            if done_count >= num_trials:
                break

            while len(running) < self.n_concurrent_jobs and done_count < num_trials:
                job = submit_one()
                running.append(job)
                print(f"[submitit] submitted job {job.job_id}", flush=True)

            if not running:
                break
            time.sleep(helper.POLL_S)

        # Wait for remaining jobs
        for j in running:
            try:
                j.result()
            except Exception as e:
                print(f"[submitit] job {j.job_id} finished with error: {e}", flush=True)
        print("[submitit] all jobs finished.", flush=True)