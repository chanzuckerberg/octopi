from octopi.entry_points import common
from octopi.utils import parsers, io
import copick, argparse, pprint, os
from typing import List, Tuple
import multiprocess as mp

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
    from octopi.workflows import localize

    # Run 3D Localization
    localize(
        copick_config_path, voxel_size, seg_info, pick_user_id, pick_session_id, n_procs,
        method, filter_size, radius_min_scale, radius_max_scale, 
        run_ids = runIDs, pick_objects = pick_objects
    )

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
    overlay_root = io.remove_prefix(root.config.overlay_root)
    basepath = os.path.join(overlay_root, 'logs')
    os.makedirs(basepath, exist_ok=True)
    output_path = os.path.join(basepath, f'localize-{args.pick_user_id}_{args.pick_session_id}.yaml')
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

