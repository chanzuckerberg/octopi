from octopi.entry_points import common
from octopi.utils import parsers, io
from octopi.extract import localize
import copick, argparse, pprint, os
from typing import List, Tuple
import multiprocess as mp
from tqdm import tqdm

def pick_particles(
    copick_config_path: str,
    method: str,
    seg_info: Tuple[str, str, str],
    voxel_size: float,
    pick_session_id: str,
    pick_user_id: str,
    radius_min_scale: float,
    radius_max_scale: float,
    filter_size: float,
    pick_objects: List[str],
    runIDs: List[str],
    n_procs: int,
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
    if pick_objects is not None:
        objects = [obj for obj in objects if obj[0] in pick_objects]

    print(f'Running Localization on the Following Objects: ')
    print(', '.join([f'{obj[0]} (Label: {obj[1]})' for obj in objects]) + '\n')

    # Either Specify Input RunIDs or Run on All RunIDs
    if runIDs:
        print('Running Localization on the Following RunIDs: ' + ', '.join(runIDs) + '\n')
        run_ids = runIDs
    else:
        run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is not None]
        skipped_run_ids = [run.name for run in root.runs if run.get_voxel_spacing(voxel_size) is None]

        if skipped_run_ids:
            print(f"Warning: skipping runs with no voxel spacing {voxel_size}: {skipped_run_ids}")

    # Nprocesses shouldnt exceed computation resource or number of available runs
    n_run_ids = len(run_ids)
    n_procs = min(mp.cpu_count(), n_procs, n_run_ids)

    # Run Localization - Main Parallelization Loop
    print(f"Using {n_procs} processes to parallelize across {n_run_ids} run IDs.")
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=n_run_ids, desc="Localization", unit="run") as pbar:
            worker_func = lambda run_id: localize.process_localization(
                root.get_run(run_id),
                objects, 
                seg_info,
                method, 
                voxel_size,
                filter_size,
                radius_min_scale, 
                radius_max_scale,
                pick_session_id,
                pick_user_id
            )

            for _ in pool.imap_unordered(worker_func, run_ids, chunksize=1):
                pbar.update(1)

    

    print('Localization Complete!')

def localize_parser(parser_description, add_slurm: bool = False):
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--config", type=str, required=True, help="Path to the CoPick configuration file.")
    input_group.add_argument("--method", type=str, choices=['watershed', 'com'], default='watershed', required=False, help="Localization method to use.")
    input_group.add_argument('--seg-info', type=parsers.parse_target, required=False, default='predict,octopi,1', help='Query for the organelles segmentations (e.g., "name" or "name,user_id,session_id".).')
    input_group.add_argument("--voxel-size", type=float, default=10, required=False, help="Voxel size for localization.")
    input_group.add_argument("--runIDs", type=parsers.parse_list, default = None, required=False, help="List of runIDs to run inference on, e.g., run1,run2,run3 or [run1,run2,run3].")

    localize_group = parser.add_argument_group("Localize Arguments")
    localize_group.add_argument("--radius-min-scale", type=float, default=0.5, required=False, help="Minimum radius scale for particles.")
    localize_group.add_argument("--radius-max-scale", type=float, default=1.0, required=False, help="Maximum radius scale for particles.")
    localize_group.add_argument("--filter-size", type=int, default=10, required=False, help="Filter size for localization.")
    localize_group.add_argument("--pick-objects", type=parsers.parse_list, default=None, required=False, help="Specific Objects to Find Picks for.")
    localize_group.add_argument("--n-procs", type=int, default=8, required=False, help="Number of CPU processes to parallelize runs across. Defaults to the max number of cores available or available runs.")

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--pick-session-id", type=str, default='1', required=False, help="Session ID for the particle picks.")
    output_group.add_argument("--pick-user-id", type=str, default='octopi', required=False, help="User ID for the particle picks.")

    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'localize', gpus = 0)

    args = parser.parse_args()
    return args

# Entry point with argparse
def cli():
    
    parser_description = "Localized particles in tomograms using multiprocessing."
    args = localize_parser(parser_description)

    # Save JSON with Parameters
    root = copick.from_file(args.config)
    basepath = os.path.join(root.config.overlay_root, 'logs')
    if basepath[:8] == 'local://': basepath = basepath[8:]
    os.makedirs(basepath, exist_ok=True)
    output_path = os.path.join(
        basepath, 
        f'localize-{args.pick_user_id}_{args.pick_session_id}.yaml')
    save_parameters(args, output_path)    


    # Set multiprocessing start method
    mp.set_start_method("spawn")
    
    pick_particles(
        copick_config_path=args.config,
        method=args.method,
        seg_info=args.seg_info,
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

def save_parameters(args: argparse.Namespace, 
                    output_path: str):

    # Organize parameters into categories
    params = {
        "input": {
            "config": args.config,
            "seg_name": args.seg_info[0],
            "seg_user_id": args.seg_info[1],
            "seg_session_id": args.seg_info[2],
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
        }
    }

    # Print the parameters
    print(f"\nParameters for Localization:")
    pprint.pprint(params); print()

    # Save to YAML file
    io.save_parameters_yaml(params, output_path)

if __name__ == "__main__":
    cli()

