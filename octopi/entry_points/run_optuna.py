from octopi.entry_points import common
from octopi.utils import parsers
import rich_click as click

@click.command('model-explore', no_args_is_help=True)
# Training Arguments
@click.option('--random-seed', type=int, default=42,
              help="Random seed for reproducibility")
@common.train_parameters(octopi=True)
# Model Arguments
@click.option('--model-type', type=click.Choice(['Unet', 'AttentionUnet', 'MedNeXt', 'SegResNet'], case_sensitive=False),
              default='Unet',
              help="Model type to use for training")
# Input Arguments
@click.option('-split', '--data-split', type=str, default='0.8',
              help="Data split ratios. Either a single value (e.g., '0.8' for 80/20/0 split) or two comma-separated values (e.g., '0.7,0.1' for 70/10/20 split)")
@click.option('-vruns', '--validateRunIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of validation run IDs, e.g., run3,run4 or [run3,run4]")
@click.option('-truns', '--trainRunIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of training run IDs, e.g., run1,run2 or [run1,run2]")
@click.option('-n', '--study-name', type=str, default="model-search",
              help="Name of the Optuna/MLflow experiment")
@click.option('-alg', '--tomo-alg', type=str, default='wbp',
              help="Tomogram algorithm used for training, provide a comma-separated list of algorithms for multiple options. (e.g., 'denoised,wbp')")
@click.option('-tinfo', '--target-info', type=str, default="targets,octopi,1",
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help="Target information, e.g., 'name' or 'name,user_id,session_id'")
@click.option('-o', '--output', type=str, default='explore_results',
              help="Name of the output directory")

# Submitit Arguments
@click.option('--submitit', type=bool, default=False,
              help="Submit trials via SLURM (submitit) instead of local GPUs")
@click.option('--njobs', '-nj', type=int, default=5,
              help="Number of concurrent training jobs when using submitit")
@click.option('--compute-constraint', '-cc', type=str, default='4,16',
              help='Compute constraint for number of CPUs requested and mem-per-cpu requested. (e.g., "4,16" for 4 CPUs and 16GB per CPU)')
@click.option('--timeout', type=int, default=4,
              help="SLURM job timeout per trial when using submitit (hours)")
@common.config_parameters(single_config=False)
def cli(
    config, voxel_size, target_info, tomo_alg, study_name, 
    trainrunids, validaterunids, data_split, model_type, num_epochs, background_ratio,
    val_interval, ncache_tomos, best_metric, num_trials, random_seed, output,
    submitit, njobs, compute_constraint, timeout):
    """
    Perform model architecture search with Optuna.
    """

    print('\n🚀 Starting a new Octopi Model Architecture Search...\n')
    run_model_explore(
        config, voxel_size, target_info, tomo_alg, study_name, 
        trainrunids, validaterunids, data_split, model_type, background_ratio,
        num_epochs, val_interval, ncache_tomos, best_metric, num_trials, random_seed, output,
        submitit=submitit, njobs=njobs, compute_constraint=compute_constraint, timeout=timeout,
    )

def run_model_explore(config, voxel_size, target_info, tomo_alg, study_name, 
        trainrunids, validaterunids, data_split, model_type, background_ratio,
        num_epochs, val_interval, ncache_tomos, best_metric, num_trials, random_seed, 
        output, submitit, njobs, compute_constraint, timeout):
    """
    Run the model exploration (local GPUs or SLURM via submitit).
    """
    from octopi.pytorch.submit_search import SubmititExplorer, ExploreSubmitter
    import os

    # Parse the CoPick configuration paths
    if len(config) > 1:
        copick_configs = parsers.parse_copick_configs(config)
    else:
        copick_configs = config[0]

    # Create the model exploration directory
    os.makedirs(output, exist_ok=True)

    base_kwargs = dict(
        copick_config=copick_configs,
        target_name=target_info[0],
        target_user_id=target_info[1],
        target_session_id=target_info[2],
        tomo_algorithm=tomo_alg,
        voxel_size=voxel_size,
        model_type=model_type,
        random_seed=random_seed,
        num_epochs=num_epochs,
        num_trials=num_trials,
        trainRunIDs=trainrunids,
        validateRunIDs=validaterunids,
        ntomo_cache=ncache_tomos,
        best_metric=best_metric,
        val_interval=val_interval,
        data_split=data_split,
        background_ratio=background_ratio,
    )


    if submitit:
        timeout = timeout * 60 # convert hours to minutes
        (ncpus, cpu_mem) = parse_compute_constraint(compute_constraint)
        search = SubmititExplorer(
            n_concurrent_jobs=njobs,
            cpus_per_task=ncpus,
            mem_per_cpu=cpu_mem,
            slurm_timeout_min=timeout,
            **base_kwargs,
        )
    else:
        search = ExploreSubmitter(**base_kwargs)

    # Run the model search
    search.run_model_search(study_name, output)

def parse_compute_constraint(compute_constraint: str) -> tuple[int, int]:
    """Parse 'cpus,mem_per_cpu_gb' (e.g. '4,16') into (cpus_per_task, mem_gb_total)."""
    parts = [p.strip() for p in compute_constraint.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"compute_constraint must be 'cpus,mem_per_cpu_gb' (e.g. '4,16'), got: {compute_constraint!r}"
        )
    cpus = int(parts[0])
    mem_per_cpu_gb = int(parts[1])
    if cpus < 1 or mem_per_cpu_gb < 1:
        raise ValueError(
            f"compute_constraint cpus and mem_per_cpu_gb must be positive, got: {compute_constraint!r}"
        )
    return cpus, mem_per_cpu_gb

if __name__ == "__main__":
    cli()