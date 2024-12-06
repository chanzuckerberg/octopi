from model_explore.pytorch import localize, utils
import copick, argparse, json
import multiprocess as mp
from typing import List
from tqdm import tqdm

def pick_particles(
    copick_config_path: str,
    method: str = 'watershed',
    seg_name: str = 'segment-predict',
    seg_user_id: str = None,
    seg_session_id: str = None,
    voxel_size: str = 10,
    pick_session_id: str = '1',
    pick_user_id: str = 'monai',
    radius_min_scale: float = 0.5,
    radius_max_scale: float = 1.5,
    filter_size: float = 10,
    pick_objects: List[str] = None,
    runIDs: List[str] = None,
    n_procs: int = None,
    ):

    # Load the Copick Project
    root = copick.from_file(copick_config_path)    

    # Get objects that can be Picked
    objects = [(obj.name, obj.label, obj.radius) for obj in root.pickable_objects if obj.is_particle]

     # Verify each object has the required attributes
    for obj in objects:
        if len(obj) < 3 or not isinstance(obj[2], (float, int)):
            raise ValueError(f"Invalid object format: {obj}. Expected a tuple with (name, label, radius).")

    # Filter elements
    objects = [obj for obj in objects if obj[0] in pick_objects]

    print(f'Running Localization on the Following Objects: ')
    print(', '.join([f'{obj[0]} (Label: {obj[1]})' for obj in objects]) + '\n')

    # Either Specify Input RunIDs or Run on All RunIDs
    if runIDs:  print('Running Localization on the Following RunIDs: ' + ', '.join(runIDs) + '\n')
    run_ids = runIDs if runIDs else [run.name for run in root.runs]
    n_run_ids = len(run_ids)

    # Determine the number of processes to use
    if n_procs is None:
        n_procs = min(mp.cpu_count(), n_run_ids)
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")

    # Initialize tqdm progress bar
    with tqdm(total=n_run_ids, desc="Localization", unit="run") as pbar:
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
                    target=localize.processs_localization,
                    args=(run,  
                          objects, 
                          seg_name,
                          seg_user_id,
                          seg_session_id,
                          method, 
                          voxel_size,
                          filter_size,
                          radius_min_scale, 
                          radius_max_scale,
                          pick_session_id,
                          pick_user_id),
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

    print('Localization Complete!')

# Entry point with argparse
def cli():
    parser = argparse.ArgumentParser(description="Localized particles in tomograms using multiprocessing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    parser.add_argument("--method", type=str, choices=['watershed', 'com'], default='watershed', required=False, help="Localization method to use.")
    parser.add_argument("--seg-name", type=str, default='segment-predict', required=False, help="Name of the segmentation.")
    parser.add_argument("--seg-user-id", type=str, default=None,  required=False, help="User ID for the segmentation.")
    parser.add_argument("--seg-session-id", type=str, default=None, required=False, help="Session ID for the segmentation.")
    parser.add_argument("--voxel-size", type=float, default=10, required=False, help="Voxel size for localization.")
    parser.add_argument("--pick-session-id", type=str, default='1', required=False, help="Session ID for the particle picks.")
    parser.add_argument("--pick-user-id", type=str, default='monai', required=False, help="User ID for the particle picks.")
    parser.add_argument("--radius-min-scale", type=float, default=0.5, required=False, help="Minimum radius scale for particles.")
    parser.add_argument("--radius-max-scale", type=float, default=1.5, required=False, help="Maximum radius scale for particles.")
    parser.add_argument("--filter-size", type=float, default=10, required=False, help="Filter size for localization.")
    parser.add_argument("--pick-objects", type=utils.parse_list, default=None, required=False, help="Specific Objects to Find Picks for.")
    parser.add_argument("--runIDs", type=utils.parse_list, default = None, required=False, help="List of runIDs to run inference on, e.g., run1,run2,run3 or [run1,run2,run3].")
    parser.add_argument("--n-procs", type=int, default=None, required=False, help="Number of CPU processes to parallelize runs across. Defaults to the max number of cores available or available runs.")

    args = parser.parse_args()

    # Save JSON with Parameters
    output_json = f'localize_{args.pick_user_id}_{args.pick_session_id}.json'    
    save_parameters_json(args, output_json)    

    # Set multiprocessing start method
    mp.set_start_method("spawn")
    
    pick_particles(
        copick_config_path=args.config,
        method=args.method,
        seg_name=args.seg_name,
        seg_user_id=args.seg_user_id,
        seg_session_id=args.seg_session_id,
        voxel_size=args.voxel_size,
        pick_session_id=args.pick_session_id,
        pick_user_id=args.pick_user_id,
        radius_min_scale=args.radius_min_scale,
        radius_max_scale=args.radius_max_scale,
        filter_size=args.filter_size,
        runIDs=args.runIDs,
        pick_objects=args.pick_objects,
        n_procs=args.n_procs,
    )

def save_parameters_json(args: argparse.Namespace, 
                         output_path: str):

    # Organize parameters into categories
    params = {
        "input": {
            "config": args.config,
            "seg_name": args.seg_name,
            "seg_user_id": args.seg_user_id,
            "seg_session_id": args.seg_session_id,
            "voxel_size": args.voxel_size
        },
        "output": {
            "pick_session_id": args.pick_session_id,
            "pick_user_id": args.pick_user_id
        },
        "parameters": {
            "method": args.method,
            "radius_min_scale": args.radius_min_scale,
            "radius_max_scale": args.radius_max_scale,
            "filter_size": args.filter_size,
            "runIDs": args.runIDs
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)                         

if __name__ == "__main__":
    cli()

# def time_pick_particles():
#     import json, time

#     # Set multiprocessing start method
#     mp.set_start_method("spawn")

#     copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"  # Replace with your actual path
#     n_procs_list = [1, 4, 8, 16, 32]  # Adjust based on your needs
#     n_procs_list = [32, 16, 8, 4, 1]
#     timing_results = {}

#     session_id = 1
#     for n_procs in n_procs_list:
#         print(f"Testing with {n_procs} processes...")
#         start_time = time.time()
#         pick_particles(
#             copick_config_path=copick_config_path,
#             pick_session_id=str(session_id),
#             n_procs=n_procs
#         )
#         elapsed_time = time.time() - start_time
#         timing_results[n_procs] = elapsed_time
#         print(f"Elapsed time with {n_procs} processes: {elapsed_time:.2f} seconds")

#         session_id +=1 

#     # Save timing results to a JSON file
#     with open("timing_results.json", "w") as f:
#         json.dump(timing_results, f, indent=4)

#     print("Timing results saved to 'timing_results.json'")

# if __name__ == "__main__":
#     time_pick_particles()