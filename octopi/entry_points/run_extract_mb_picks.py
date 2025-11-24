from typing import List, Tuple, Optional
from octopi.utils import parsers
import rich_click as click

def extract_membrane_bound_picks(
    config: str,
    voxel_size: float,
    distance_threshold: float,
    picks_info: Tuple[str, str, str],
    organelle_info: Tuple[str, str, str],
    membrane_info: Tuple[str, str, str],
    save_user_id: str,
    save_session_id: str,
    runIDs: List[str],
    n_procs: int = None
    ):  
    from octopi.extract import membranebound_extract as extract
    import multiprocess as mp
    from tqdm import tqdm
    import copick

    # Load Copick Project for Writing 
    root = copick.from_file(config) 
    
    # Either Specify Input RunIDs or Run on All RunIDs
    if runIDs:  print('Extracting Membrane Bound Proteins on the Following RunIDs: ', runIDs)
    run_ids = runIDs if runIDs else [run.name for run in root.runs]
    n_run_ids = len(run_ids)    

    # Determine the number of processes to use
    if n_procs is None:
        n_procs = min(mp.cpu_count(), n_run_ids)
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")   
    
    # Run Membrane-Protein Isolation - Main Parallelization Loop
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=n_run_ids, desc="Membrane-Protein Isolation", unit="run") as pbar:
            worker_func = lambda run_id: extract.process_membrane_bound_extract(
                root.get_run(run_id),  
                voxel_size, 
                picks_info, 
                membrane_info,
                organelle_info,
                save_user_id, 
                save_session_id,
                distance_threshold
            )

            for _ in pool.imap_unordered(worker_func, run_ids, chunksize=1):
                pbar.update(1)

    print('Extraction of Membrane-Bound Proteins Complete!')


def save_parameters(config: str,
                    voxel_size: float,
                    picks_info: tuple,
                    membrane_info: tuple,
                    organelle_info: tuple,
                    save_user_id: str,
                    save_session_id: str,
                    distance_threshold: float,
                    runIDs: list,
                    output_path: str):
    import octopi.utils.io as io
    import pprint

    params_dict = {
        "input": {
            "config": config,
            "voxel_size": voxel_size,
            "picks_info": picks_info,
            "membrane_info": membrane_info,
            "organelle_info": organelle_info
        },
        "output": {
            "save_user_id": save_user_id,
            "save_session_id": save_session_id
        },
        "parameters": {
            "distance_threshold": distance_threshold,
            "runIDs": runIDs
        }
    }

    # Print the parameters
    print(f"\nParameters for Extraction of Membrane-Bound Picks:")
    pprint.pprint(params_dict); print()

    # Save parameters to YAML file
    io.save_parameters_yaml(params_dict, output_path)


@click.command('membrane-extract', help="Extract membrane-bound picks based on proximity to segmentation")
# Output Arguments
@click.option('--save-session-id', type=str, required=True,
              help="Session ID to save the new picks")
@click.option('--save-user-id', type=str, default=None,
              help="User ID to save the new picks (defaults to picks user ID)")
# Parameters
@click.option('--n-procs', type=int, default=None,
              help="Number of processes to use (defaults to CPU count)")
@click.option('--distance-threshold', type=float, default=10,
              help="Distance threshold for membrane proximity")
# Input Arguments
@click.option('--runIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of run IDs to process")
@click.option('--organelle-info', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_target(value) if value else None,
              help='Query for the organelles segmentations (e.g., "name" or "name,user_id,session_id")')
@click.option('--membrane-info', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_target(value) if value else None,
              help='Query for the membrane segmentation (e.g., "name" or "name,user_id,session_id")')
@click.option('--picks-info', type=str, required=True,
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help='Query for the picks (e.g., "name" or "name,user_id,session_id")')
@click.option('-vs', '--voxel-size', type=float, default=10,
              help="Voxel size")
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the configuration file")
def cli(config, voxel_size, picks_info, membrane_info, organelle_info, runIDs,
        distance_threshold, n_procs,
        save_user_id, save_session_id):
    """
    CLI entry point for extracting membrane-bound picks.
    """

    run_mb_extract(
        config, voxel_size,
        picks_info, membrane_info, organelle_info,
        runIDs, distance_threshold, n_procs,
        save_user_id, save_session_id
    )


def run_mb_extract(
    config, voxel_size, picks_info, membrane_info, 
    organelle_info, runIDs, distance_threshold, n_procs,
    save_user_id, save_session_id):
    
    # Default save_user_id to picks_info user_id if not specified
    if save_user_id is None: 
        save_user_id = picks_info[1]

    # Save parameters
    output_yaml = f'membrane-extract_{save_user_id}_{save_session_id}.yaml'
    save_parameters(
        config=config,
        voxel_size=voxel_size,
        picks_info=picks_info,
        membrane_info=membrane_info,
        organelle_info=organelle_info,
        save_user_id=save_user_id,
        save_session_id=save_session_id,
        distance_threshold=distance_threshold,
        runIDs=runIDs,
        output_path=output_yaml
    )

    extract_membrane_bound_picks(
        config=config,
        voxel_size=voxel_size,
        distance_threshold=distance_threshold,
        picks_info=picks_info,
        membrane_info=membrane_info,
        organelle_info=organelle_info,
        save_user_id=save_user_id,
        save_session_id=save_session_id,
        runIDs=runIDs,
        n_procs=n_procs,
    )


if __name__ == "__main__":
    cli()