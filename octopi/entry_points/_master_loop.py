"""
Internal module for running the master monitoring loop.

This module is not intended to be called directly by users. It is invoked
by the SLURM master job script to monitor and manage worker submissions.
"""

import time
import os
import subprocess
import argparse
from octopi.utils import slurm_monitor, parsers, config
from octopi.pytorch.model_search_submitter import ModelSearchSubmit
from octopi.entry_points import run_optuna_master
import optuna


def master_monitoring_loop(args):
    """
    Main monitoring loop for the master job.

    This function:
    1. Creates the Optuna study
    2. Submits initial batch of workers
    3. Continuously monitors progress
    4. Submits new workers as needed
    5. Handles failures and retries
    6. Exits when complete or conditions met

    Args:
        args: Parsed arguments from run_optuna_master parser
    """

    # Setup
    model_type = args.model_type
    results_dir = f'explore_results_{model_type}'
    os.makedirs(results_dir, exist_ok=True)

    db_path = os.path.abspath(f"{results_dir}/trials.db")
    study_name = f"model-search-{model_type}"
    log_file = f"{results_dir}/master.log"

    # Initialize logging
    with open(log_file, 'w') as f:
        f.write(f"Master job started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Study name: {study_name}\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Total trials: {args.num_trials}\n")
        f.write(f"Max parallel workers: {args.max_parallel_workers}\n")
        f.write(f"Worker time limit: {args.worker_time_limit}\n")
        f.write(f"Monitor interval: {args.monitor_interval}s\n")
        f.write("=" * 70 + "\n\n")

    print(f"Master job starting...")
    print(f"Results directory: {results_dir}")
    print(f"Log file: {log_file}")

    # Parse CoPick configuration
    if len(args.config) > 1:
        copick_configs = parsers.parse_copick_configs(args.config)
    else:
        copick_configs = args.config[0]

    # Save parameters
    run_optuna.save_parameters(args, f'{results_dir}/octopi.yaml')

    # Create ModelSearchSubmit instance
    search = ModelSearchSubmit(
        copick_config=copick_configs,
        target_name=args.target_info[0],
        target_user_id=args.target_info[1],
        target_session_id=args.target_info[2],
        tomo_algorithm=args.tomo_alg,
        voxel_size=args.voxel_size,
        Nclass=args.Nclass,
        model_type=args.model_type,
        mlflow_experiment_name=args.mlflow_experiment_name,
        random_seed=args.random_seed,
        num_epochs=args.num_epochs,
        num_trials=args.num_trials,
        trainRunIDs=args.trainRunIDs,
        validateRunIDs=args.validateRunIDs,
        tomo_batch_size=args.tomo_batch_size,
        best_metric=args.best_metric,
        val_interval=args.val_interval,
        data_split=args.data_split
    )

    # Create the Optuna study
    print(f"Creating Optuna study: {study_name}")

    # Set up MLflow if available
    try:
        tracking_uri = config.mlflow_setup()
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
    except Exception as e:
        print(f'Warning: Failed to set up MLflow tracking: {e}')

    # Create storage with improved concurrency settings
    storage_url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
        engine_kwargs={
            "connect_args": {
                "timeout": 300,
                "check_same_thread": False
            },
            "pool_pre_ping": True
        }
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=search._get_optuna_sampler(),
        load_if_exists=True,
        pruner=search._get_optuna_pruner()
    )

    print(f"Study created successfully\n")

    # Initialize worker tracker
    tracker = slurm_monitor.WorkerTracker()
    start_time = time.time()

    # Submit initial batch of workers
    initial_workers = min(args.max_parallel_workers, args.num_trials)
    print(f"Submitting initial batch of {initial_workers} workers...")

    for i in range(initial_workers):
        job_id = submit_worker(args, db_path, study_name, results_dir)
        if job_id:
            tracker.add_worker(job_id, time.time())
            print(f"  Worker {i+1}/{initial_workers}: Job ID {job_id}")

    print(f"\nMaster monitoring loop started")
    print(f"Check progress in: {log_file}\n")

    # Main monitoring loop
    iteration = 0
    while True:
        iteration += 1

        # Update worker statuses from SLURM
        active_job_ids = list(tracker.active_workers.keys())
        if active_job_ids:
            slurm_statuses = slurm_monitor.query_slurm_status(active_job_ids)

            for job_id, status in slurm_statuses.items():
                tracker.update_status(job_id, status)

                # Handle completed workers
                if status in ['COMPLETED']:
                    tracker.mark_completed(job_id)

                # Handle failed workers
                elif status in ['FAILED', 'TIMEOUT', 'CANCELLED', 'NOT_FOUND']:
                    metadata = tracker.active_workers.get(job_id)
                    if metadata and metadata.retry_count < args.max_worker_retries:
                        # Retry
                        print(f"Retrying failed worker {job_id} (attempt {metadata.retry_count + 2})")
                        new_job_id = submit_worker(args, db_path, study_name, results_dir)
                        if new_job_id:
                            tracker.retry_worker(job_id, new_job_id)
                            with open(log_file, 'a') as f:
                                f.write(f"Retried worker {job_id} -> {new_job_id} (retry {metadata.retry_count + 1})\n")
                    else:
                        # Mark as permanently failed
                        tracker.mark_failed(job_id)
                        with open(log_file, 'a') as f:
                            f.write(f"Worker {job_id} permanently failed after {args.max_worker_retries} retries\n")

        # Get Optuna progress
        progress = slurm_monitor.query_optuna_progress(db_path, study_name)

        # Log progress
        slurm_monitor.log_progress(log_file, tracker, progress, start_time, args.num_trials)

        # Check exit conditions
        should_exit, reason = check_exit_conditions(
            tracker, progress, start_time, args, results_dir
        )

        if should_exit:
            print(f"\n{'='*70}")
            print(f"Master exiting: {reason}")
            print(f"{'='*70}\n")
            print_final_summary(tracker, progress, start_time, args.num_trials, log_file)
            break

        # Maintain worker pool - submit new workers as needed
        trials_done = progress['COMPLETE'] + progress['FAIL'] + progress['PRUNED']
        trials_in_progress = progress['RUNNING'] + tracker.get_active_count()
        trials_remaining = args.num_trials - trials_done - trials_in_progress

        # Calculate how many workers to submit
        active_count = tracker.get_active_count()
        slots_available = args.max_parallel_workers - active_count
        workers_to_submit = min(slots_available, trials_remaining)

        if workers_to_submit > 0:
            print(f"Submitting {workers_to_submit} new workers (iteration {iteration})")
            for i in range(workers_to_submit):
                job_id = submit_worker(args, db_path, study_name, results_dir)
                if job_id:
                    tracker.add_worker(job_id, time.time())
                    with open(log_file, 'a') as f:
                        f.write(f"Submitted worker {job_id}\n")

        # Sleep before next iteration
        time.sleep(args.monitor_interval)


def submit_worker(args, db_path: str, study_name: str, results_dir: str) -> str:
    """
    Submit a single worker job to SLURM.

    Args:
        args: Parsed arguments
        db_path: Path to Optuna database
        study_name: Name of the study
        results_dir: Results directory

    Returns:
        SLURM job ID, or None if submission failed
    """
    from octopi.utils import submit_slurm
    import tempfile

    # Build worker command
    cmd_parts = ['octopi', 'model-explore', '--worker-mode']
    cmd_parts.extend(['--study-name', study_name])
    cmd_parts.extend(['--study-db-path', db_path])

    # Add configuration
    if isinstance(args.config, list):
        for cfg in args.config:
            cmd_parts.extend(['--config', cfg])
    else:
        cmd_parts.extend(['--config', args.config])

    # Add other arguments
    cmd_parts.extend(['--target-info', f"{args.target_info[0]},{args.target_info[1]},{args.target_info[2]}"])
    cmd_parts.extend(['--tomo-alg', args.tomo_alg])
    cmd_parts.extend(['--voxel-size', str(args.voxel_size)])
    cmd_parts.extend(['--model-type', args.model_type])
    cmd_parts.extend(['--Nclass', str(args.Nclass)])
    cmd_parts.extend(['--mlflow-experiment-name', args.mlflow_experiment_name])
    cmd_parts.extend(['--random-seed', str(args.random_seed)])
    cmd_parts.extend(['--num-epochs', str(args.num_epochs)])
    cmd_parts.extend(['--num-trials', str(args.num_trials)])
    cmd_parts.extend(['--tomo-batch-size', str(args.tomo_batch_size)])
    cmd_parts.extend(['--best-metric', args.best_metric])
    cmd_parts.extend(['--val-interval', str(args.val_interval)])
    cmd_parts.extend(['--data-split', args.data_split])

    if args.trainRunIDs:
        cmd_parts.extend(['--trainRunIDs', ','.join(args.trainRunIDs)])
    if args.validateRunIDs:
        cmd_parts.extend(['--validateRunIDs', ','.join(args.validateRunIDs)])

    command = ' '.join(cmd_parts)

    # Create temporary worker script
    worker_script = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, dir=results_dir)
    worker_script_path = worker_script.name
    worker_script.close()

    # Generate SLURM script
    conda_env = getattr(args, 'conda_env', '/hpc/projects/group.czii/conda_environments/pyUNET/')
    gpu_constraint = getattr(args, 'gpu_constraint', 'h100')

    submit_slurm.create_shellsubmit(
        job_name=f"optuna-worker-{args.model_type}",
        output_file=f"{results_dir}/worker-%j.log",
        shell_name=worker_script_path,
        conda_path=conda_env,
        command=command,
        num_gpus=1,
        gpu_constraint=gpu_constraint,
        time_limit=args.worker_time_limit
    )

    # Submit to SLURM
    try:
        result = subprocess.run(
            ['sbatch', worker_script_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            job_id = slurm_monitor.parse_slurm_job_id(result.stdout)
            return job_id
        else:
            print(f"Warning: sbatch failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Warning: Failed to submit worker: {e}")
        return None


def check_exit_conditions(tracker, progress, start_time, args, results_dir):
    """
    Check if the master should exit.

    Returns:
        (should_exit: bool, reason: str)
    """
    # Check for sentinel file
    if os.path.exists('.stop_optuna_master'):
        return True, "User stop signal received (.stop_optuna_master file found)"

    # All trials complete or failed
    trials_done = progress['COMPLETE'] + progress['FAIL'] + progress['PRUNED']
    if trials_done >= args.num_trials:
        return True, f"All trials complete ({trials_done}/{args.num_trials})"

    # Approaching time limit
    elapsed = time.time() - start_time
    time_limit_seconds = parse_time_limit(args.master_time_limit)
    if elapsed > (time_limit_seconds * 0.9):
        return True, f"Approaching master time limit (90% of {args.master_time_limit} elapsed)"

    # Error threshold exceeded
    if trials_done > 0:
        failure_rate = progress['FAIL'] / trials_done
        if failure_rate > args.error_threshold:
            return True, f"Error threshold exceeded: {failure_rate:.1%} > {args.error_threshold:.1%}"

    return False, None


def parse_time_limit(time_str):
    """Parse HH:MM:SS time limit to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    return 72 * 3600  # Default 72 hours


def print_final_summary(tracker, progress, start_time, total_trials, log_file):
    """Print final summary of the optimization run."""
    elapsed = time.time() - start_time
    worker_summary = tracker.get_summary()

    summary = f"""
{'='*70}
FINAL SUMMARY
{'='*70}

Trials:
  Completed: {progress['COMPLETE']}/{total_trials}
  Failed: {progress['FAIL']}
  Pruned: {progress['PRUNED']}
  Running: {progress['RUNNING']}

Workers:
  Total submitted: {worker_summary['completed'] + worker_summary['failed'] + worker_summary['active']}
  Completed: {worker_summary['completed']}
  Failed: {worker_summary['failed']}
  Still active: {worker_summary['active']}

Time:
  Total elapsed: {slurm_monitor.format_time_remaining(elapsed)}

{'='*70}
"""

    print(summary)

    with open(log_file, 'a') as f:
        f.write(summary)
        f.write(f"\nMaster job completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def cli():
    """
    Internal CLI entry point for the master monitoring loop.
    This is called by the SLURM master job script.
    """
    # Use the same parser as the main entry point
    description = "Internal: Run the master monitoring loop"
    args = run_optuna_master.optuna_master_parser(description, add_slurm=False)

    # Run the monitoring loop
    master_monitoring_loop(args)


if __name__ == "__main__":
    cli()
