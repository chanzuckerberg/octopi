from model_explore.pytorch import midpoint_extract, utils
import copick, subprocess, glob, os, click, json
from typing import List, Tuple, Optional
import argparse, json, pprint
import multiprocess as mp
from tqdm import tqdm
import numpy as np

def extract_midpoint(
    config: str,
    voxel_size: float,
    picks_info: Tuple[str, str, str],
    organelle_info: Tuple[str, str, str],
    distance_min: float,
    distance_max: float,
    distance_threshold: float,
    save_user_id: str,
    save_session_id: str,
    runIDs: List[str],
    n_procs: int = None
    ):  

    # Load Copick Project for Writing 
    root = copick.from_file( config ) 
    
    # Either Specify Input RunIDs or Run on All RunIDs
    if runIDs:  print('Extracting Midpoints on the Following RunIDs: ', runIDs)
    run_ids = runIDs if runIDs else [run.name for run in root.runs]
    n_run_ids = len(run_ids)   

    # Determine the number of processes to use
    if n_procs is None:
        n_procs = min(mp.cpu_count(), n_run_ids)
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")   

    # Initialize tqdm progress bar
    with tqdm(total=n_run_ids, desc="Mid-Point SuperComplex Extraction", unit="run") as pbar:
        for _iz in range(0, n_run_ids, n_procs):

            start_idx = _iz
            end_idx = min(_iz + n_procs, n_run_ids)  # Ensure end_idx does not exceed n_run_ids
            print(f"\nProcessing runIDs from {start_idx} -> {end_idx } (out of {n_run_ids})")

            processes = []                
            for _in in range(n_procs):
                _iz_this = _iz + _in
                if _iz_this >= n_run_ids:
                    break
                run_id = run_ids[_iz_this]
                run = root.get_run(run_id)
                p = mp.Process(
                    target=midpoint_extract.process_midpoint_extract,
                    args=(run,  
                          voxel_size, 
                          picks_info, 
                          organelle_info,
                          distance_min,
                          distance_max,
                          distance_threshold,
                          save_user_id, 
                          save_session_id)
                )
                processes.append(p)

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            for p in processes:
                p.close()

            # Update tqdm progress bar
            pbar.update(len(processes))

    print('Extraction of Midpoints Complete!')        

def cli():
    parser = argparse.ArgumentParser(description='Extract membrane-bound picks based on proximity to segmentation.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--voxel-size', type=float, required=False, default=10, help='Voxel size.')
    parser.add_argument('--picks-info', type=utils.parse_target, required=True, help='Query for the picks (e.g., "name" or "name,user_id,session_id".).')
    parser.add_argument('--organelle-info', type=utils.parse_target, required=False, help='Query for the organelles segmentations (e.g., "name" or "name,user_id,session_id".).')
    parser.add_argument('--distance-min', type=float, required=False, default=10, help='Minimum distance for valid nearest neighbors.') 
    parser.add_argument('--distance-max', type=float, required=False, default=70, help='Maximum distance for valid nearest neighbors.')
    parser.add_argument('--distance-threshold', type=float, required=False, default=25, help='Distance threshold for picks to associated organelles.')
    parser.add_argument('--save-user-id', type=str, required=False, default=None, help='User ID to save the new picks.')
    parser.add_argument('--save-session-id', type=str, required=True, help='Session ID to save the new picks.')
    parser.add_argument('--runIDs', type=utils.parse_list, required=False, help='List of run IDs to process.')
    parser.add_argument('--n-procs', type=int, required=False, default=None, help='Number of processes to use.')

    args = parser.parse_args()

    # Increment session ID for the second class
    if args.save_user_id is None: 
        args.save_user_id = args.picks_user_id

    # Save JSON with Parameters
    output_json = f'midpoint-extract_{args.save_user_id}_{args.save_session_id}.json'        
    save_parameters_json(args, output_json)

    extract_midpoint(
        config=args.config,
        voxel_size=args.voxel_size,
        picks_info=args.picks_info,
        organelle_info=args.organelle_info,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        distance_threshold=args.distance_threshold,
        save_user_id=args.save_user_id,
        save_session_id=args.save_session_id,
        runIDs=args.runIDs,
        n_procs=args.n_procs,
    )

def save_parameters_json(args: argparse.Namespace, 
                         output_path: str):

    params_dict = {
        "input": {
            k: getattr(args, k) for k in [
                "config", "voxel_size", "picks_info", 
                "organelle_info"
            ]
        },
        "output": {
            k: getattr(args, k) for k in ["save_user_id", "save_session_id"]
        },
        "parameters": {
            k: getattr(args, k) for k in ["distance_min", "distance_max", "distance_threshold", "runIDs"]
        }
    }

    # Print the parameters
    print(f"\nParameters for Extraction of Membrane-Bound Picks:")
    pprint.pprint(params_dict); print()

    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=4)    

if __name__ == "__main__":
    cli()