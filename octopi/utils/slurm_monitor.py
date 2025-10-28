"""
Utilities for monitoring SLURM jobs and Optuna studies.
Used by the master job to track worker progress and manage job submissions.
"""

import subprocess
import sqlite3
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WorkerMetadata:
    """Metadata for a single worker job."""
    job_id: str
    submit_time: float
    retry_count: int = 0
    status: str = 'PENDING'
    trial_number: Optional[int] = None


class WorkerTracker:
    """
    Track active, completed, and failed worker jobs.
    """

    def __init__(self):
        self.active_workers: Dict[str, WorkerMetadata] = {}
        self.completed_workers: List[str] = []
        self.failed_workers: List[str] = []

    def add_worker(self, job_id: str, submit_time: float, trial_number: Optional[int] = None):
        """Add a new worker to the active tracking list."""
        self.active_workers[job_id] = WorkerMetadata(
            job_id=job_id,
            submit_time=submit_time,
            trial_number=trial_number,
            retry_count=0,
            status='PENDING'
        )

    def update_status(self, job_id: str, status: str):
        """Update the status of an active worker."""
        if job_id in self.active_workers:
            self.active_workers[job_id].status = status

    def mark_completed(self, job_id: str):
        """Mark a worker as completed and remove from active list."""
        if job_id in self.active_workers:
            self.completed_workers.append(job_id)
            del self.active_workers[job_id]

    def mark_failed(self, job_id: str):
        """Mark a worker as permanently failed and remove from active list."""
        if job_id in self.active_workers:
            self.failed_workers.append(job_id)
            del self.active_workers[job_id]

    def retry_worker(self, old_job_id: str, new_job_id: str):
        """
        Replace a failed worker with a retry.
        Increments retry count and updates job ID.
        """
        if old_job_id in self.active_workers:
            old_metadata = self.active_workers[old_job_id]
            self.active_workers[new_job_id] = WorkerMetadata(
                job_id=new_job_id,
                submit_time=time.time(),
                trial_number=old_metadata.trial_number,
                retry_count=old_metadata.retry_count + 1,
                status='PENDING'
            )
            del self.active_workers[old_job_id]

    def get_active_count(self) -> int:
        """Get the number of active workers (PENDING + RUNNING)."""
        return len([w for w in self.active_workers.values()
                   if w.status in ['PENDING', 'RUNNING']])

    def get_running_count(self) -> int:
        """Get the number of running workers."""
        return len([w for w in self.active_workers.values() if w.status == 'RUNNING'])

    def get_pending_count(self) -> int:
        """Get the number of pending workers."""
        return len([w for w in self.active_workers.values() if w.status == 'PENDING'])

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of worker states."""
        return {
            'active': len(self.active_workers),
            'running': self.get_running_count(),
            'pending': self.get_pending_count(),
            'completed': len(self.completed_workers),
            'failed': len(self.failed_workers)
        }


def query_slurm_status(job_ids: List[str]) -> Dict[str, str]:
    """
    Query SLURM for job statuses.

    Args:
        job_ids: List of SLURM job IDs to query

    Returns:
        Dictionary mapping job_id to status string
        Possible statuses: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, NOT_FOUND
    """
    if not job_ids:
        return {}

    status_map = {}

    # Try squeue first (for active jobs)
    try:
        cmd = f"squeue -j {','.join(job_ids)} --format='%i,%T' --noheader"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)

        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    job_id, state = parts
                    status_map[job_id.strip()] = state.strip()
    except Exception as e:
        print(f"Warning: squeue query failed: {e}")

    # Check sacct for jobs not found in squeue (completed, failed, etc.)
    missing_jobs = [jid for jid in job_ids if jid not in status_map]
    if missing_jobs:
        try:
            cmd = f"sacct -j {','.join(missing_jobs)} --format=JobID,State --noheader --parsable2"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)

            for line in result.stdout.strip().split('\n'):
                if line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        job_id_raw, state = parts[0].strip(), parts[1].strip()
                        # sacct includes sub-jobs like "12345.batch", we want just "12345"
                        job_id = job_id_raw.split('.')[0]
                        if job_id in missing_jobs and job_id not in status_map:
                            status_map[job_id] = state
        except Exception as e:
            print(f"Warning: sacct query failed: {e}")

    # Mark any still-missing jobs as NOT_FOUND
    for job_id in job_ids:
        if job_id not in status_map:
            status_map[job_id] = 'NOT_FOUND'

    return status_map


def query_optuna_progress(db_path: str, study_name: str) -> Dict[str, int]:
    """
    Query Optuna database for trial progress.

    Args:
        db_path: Path to SQLite database
        study_name: Name of the Optuna study

    Returns:
        Dictionary with counts: {'COMPLETE': N, 'RUNNING': M, 'FAIL': X, ...}
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Get study ID
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            print(f"Warning: Study '{study_name}' not found in database")
            conn.close()
            return {'COMPLETE': 0, 'RUNNING': 0, 'FAIL': 0, 'PRUNED': 0, 'WAITING': 0}

        study_id = result[0]

        # Get trial counts by state
        cursor.execute("""
            SELECT state, COUNT(*)
            FROM trials
            WHERE study_id = ?
            GROUP BY state
        """, (study_id,))

        results = {
            'COMPLETE': 0,
            'RUNNING': 0,
            'FAIL': 0,
            'PRUNED': 0,
            'WAITING': 0
        }

        for state_code, count in cursor.fetchall():
            # Optuna stores states as strings, not integers
            # Handle both string and integer state codes for compatibility
            if isinstance(state_code, str):
                # State is already a string like "FAIL", "PRUNED", "COMPLETE"
                state_name = state_code
            else:
                # Fallback for integer state codes (if older Optuna versions use them)
                state_names = {0: 'RUNNING', 1: 'COMPLETE', 2: 'PRUNED', 3: 'FAIL', 4: 'WAITING'}
                state_name = state_names.get(state_code, f'UNKNOWN_{state_code}')

            # Only store if it's a recognized state
            if state_name in results:
                results[state_name] = count

        conn.close()
        return results

    except Exception as e:
        print(f"Warning: Failed to query Optuna database: {e}")
        return {'COMPLETE': 0, 'RUNNING': 0, 'FAIL': 0, 'PRUNED': 0, 'WAITING': 0}


def parse_slurm_job_id(sbatch_output: str) -> Optional[str]:
    """
    Parse SLURM job ID from sbatch output.

    Args:
        sbatch_output: Output from sbatch command

    Returns:
        Job ID string, or None if parsing fails

    Example:
        "Submitted batch job 12345" -> "12345"
    """
    try:
        # sbatch typically outputs: "Submitted batch job 12345"
        parts = sbatch_output.strip().split()
        if len(parts) >= 4 and parts[0] == "Submitted":
            return parts[-1]
    except:
        pass
    return None


def format_time_remaining(seconds: float) -> str:
    """
    Format seconds into human-readable time estimate.

    Args:
        seconds: Number of seconds

    Returns:
        Formatted string like "2.5h" or "45m" or "30s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    else:
        return f"{seconds / 3600:.1f}h"


def estimate_time_remaining(trials_complete: int, trials_total: int,
                           elapsed_seconds: float) -> Optional[float]:
    """
    Estimate remaining time based on progress so far.

    Args:
        trials_complete: Number of completed trials
        trials_total: Total number of trials
        elapsed_seconds: Time elapsed since start

    Returns:
        Estimated seconds remaining, or None if cannot estimate
    """
    if trials_complete == 0:
        return None

    avg_time_per_trial = elapsed_seconds / trials_complete
    trials_remaining = trials_total - trials_complete
    return avg_time_per_trial * trials_remaining


def log_progress(log_file: str, tracker: WorkerTracker, progress: Dict[str, int],
                start_time: float, total_trials: int):
    """
    Log progress to file and stdout.

    Args:
        log_file: Path to log file
        tracker: WorkerTracker instance
        progress: Optuna progress dictionary
        start_time: Start timestamp
        total_trials: Total number of trials
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - start_time

    # Calculate metrics
    completed = progress['COMPLETE']
    running = progress['RUNNING']
    failed = progress['FAIL']
    pruned = progress['PRUNED']

    worker_summary = tracker.get_summary()

    # Estimate time remaining
    time_remaining_sec = estimate_time_remaining(completed, total_trials, elapsed)
    if time_remaining_sec:
        time_remaining_str = format_time_remaining(time_remaining_sec)
    else:
        time_remaining_str = "unknown"

    # Create log message
    message = (
        f"[{timestamp}] "
        f"Trials: {completed}/{total_trials} complete "
        f"({running} running, {failed} failed, {pruned} pruned) | "
        f"Workers: {worker_summary['running']} running, {worker_summary['pending']} pending | "
        f"Est. remaining: {time_remaining_str}"
    )

    # Write to log file
    with open(log_file, 'a') as f:
        f.write(message + '\n')

    # Print to stdout
    print(message)
