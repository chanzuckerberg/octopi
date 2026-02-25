from typing import List, Tuple, Optional
from octopi.utils import parsers
import rich_click as click

def extract_membrane_bound_picks(
    config: str,
    voxel_size: float,
    distance_threshold: float,
    picks_info: Tuple[str, str, str],
    seg_info: Tuple[str, str, str],
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
                seg_info,
                save_user_id, 
                save_session_id,
                distance_threshold
            )

            for _ in pool.imap_unordered(worker_func, run_ids, chunksize=1):
                pbar.update(1)

    print('Extraction of Membrane-Bound Proteins Complete!')


def save_parameters( params_dict: dict, output_path: str ):
    import octopi.utils.io as io
    import pprint

    # Print the parameters
    print(f"\nParameters for Extraction of Membrane-Bound Picks:")
    pprint.pprint(params_dict); print()

    # Save parameters to YAML file
    io.save_parameters_yaml(params_dict, output_path)


@click.command('membrane-extract', no_args_is_help=True)
# Output Arguments
@click.option('-ssid','--save-session-id', type=str, required=True,
              help="Session ID to save the new picks")
@click.option('-suid','--save-user-id', type=str, default=None,
              help="User ID to save the new picks (defaults to picks user ID)")
# Parameters
@click.option('-np', '--n-procs', type=int, default=None,
              help="Number of processes to use (defaults to CPU count)")
@click.option('-t', '--threshold', type=str, default="1,10",
              help="Distance threshold for membrane proximity in Voxels (provide the min and max as 'min,max' if only one value is provided, it is used as max with min=1)",)
# Input Arguments
@click.option('-rids','--runIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of run IDs to process")
@click.option('-si','--seg-info', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_target(value) if value else None,
              help='Query for the membrane segmentation (e.g., "name" or "name,user_id,session_id")')
@click.option('-pi','--picks-info', type=str, required=True,
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help='Query for the picks (e.g., "name" or "name,user_id,session_id")')
@click.option('-vs', '--voxel-size', type=float, default=10,
              help="Voxel size")
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the configuration file")
def cli(config, voxel_size, picks_info, seg_info, runids,
        threshold, n_procs,
        save_user_id, save_session_id):
    """
    Extract membrane-bound picks based on proximity to organelle or membrane segmentation.
    
    This command separates particle picks into membrane-proximal and membrane-distal 
    classes by computing the minimum Euclidean distance (in voxel units) between 
    each particle coordinate and the provided segmentation mask. Picks whose 
    distance falls within the specified threshold range are classified as 
    membrane-bound; all others are classified as non-membrane-bound.

    If the original picks contain identity rotations, orientations for 
    membrane-bound particles are automatically estimated based on the vector 
    from the segmented organelle center of mass to the particle coordinate.

    The resulting picks are written back to the CoPick project:

    • Membrane-bound picks → saved under the provided session ID  

    • Non-membrane-bound picks → saved under (session ID + 1)

    All distances are interpreted in voxel units. Coordinates are converted 
    to physical units using the provided voxel size before saving.

    \b
    Examples:

    # Extract membrane-bound picks with default distance threshold (1–10 voxels)
    octopi membrane-extract -c config.json \\
        --picks-info predictions,octopi,1 \\
        --seg-info membrane,membrain-seg,1 \\
        --save-user-id octopi \\
        --save-session-id 1

    # Use a custom distance range (2–6 voxels)
    octopi membrane-extract -c config.json \\
        --picks-info predictions,octopi,1 \\
        --seg-info membrane,membrain-seg,1 \\
        --threshold 2,6 \\
        --save-user-id octopi \\
        --save-session-id 3
    """

    run_mb_extract(
        config, voxel_size,
        picks_info, seg_info,
        runids, threshold, n_procs,
        save_user_id, save_session_id
    )


def run_mb_extract(
    config, voxel_size, picks_info, seg_info, 
    runIDs, threshold, n_procs,
    save_user_id, save_session_id):
    import octopi.utils.io as io
    import copick, os

    # Parse threshold into tuple
    vals = threshold.split(',')
    if len(vals) == 1:
        threshold = (1.0, float(vals[0]))
    elif len(vals) == 2:
        threshold = (float(vals[0]), float(vals[1]))
    
    # Default save_user_id to picks_info user_id if not specified
    if save_user_id is None: 
        save_user_id = picks_info[1]

    # Save JSON with Parameters
    root = copick.from_file(config)
    overlay_root = io.remove_prefix(root.config.overlay_root)
    basepath = os.path.join(overlay_root, 'logs')
    os.makedirs(basepath, exist_ok=True)
    output_yaml = f'membrane-extract_{save_user_id}_{save_session_id}.yaml'
    output_path = os.path.join(basepath, output_yaml)        

    # Save parameters
    params_dict = {
        "input": {
            "config": config,
            "voxel_size": voxel_size,
            "picks_info": picks_info,
            "seg_info": seg_info
        },
        "output": {
            "save_user_id": save_user_id,
            "save_session_id": save_session_id
        },
        "parameters": {
            "min_distance": threshold[0],
            "max_distance": threshold[1],
            "runIDs": runIDs
        }
    }
    save_parameters(params_dict, output_path=output_path)

    extract_membrane_bound_picks(
        config=config,
        voxel_size=voxel_size,
        distance_threshold=threshold,
        picks_info=picks_info,
        seg_info=seg_info,
        save_user_id=save_user_id,
        save_session_id=save_session_id,
        runIDs=runIDs,
        n_procs=n_procs,
    )


if __name__ == "__main__":
    cli()