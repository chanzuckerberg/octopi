from octopi.utils import parsers
from typing import List, Tuple
import rich_click as click

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
    localize(copick_config_path, voxel_size, seg_info, pick_user_id, pick_session_id, n_procs,
            method, filter_size, radius_min_scale, radius_max_scale, run_ids = runIDs)


def save_parameters(seg_info: Tuple[str, str, str],
                    config: str,
                    voxel_size: float,
                    pick_session_id: str,
                    pick_user_id: str,
                    method: str,
                    radius_min_scale: float,
                    radius_max_scale: float,
                    filter_size: float,
                    output_path: str):

    import octopi.utils.io as io
    import pprint

    # Organize parameters into categories
    params = {
        "input": {
            "config": config,
            "seg_name": seg_info[0],
            "seg_user_id": seg_info[1],
            "seg_session_id": seg_info[2],
            "voxel_size": voxel_size
        },
        "output": {
            "pick_session_id": pick_session_id,
            "pick_user_id": pick_user_id
        },
        "parameters": {
            "method": method,
            "radius_min_scale": radius_min_scale,
            "radius_max_scale": radius_max_scale,
            "filter_size": filter_size,
        }
    }

    # Print the parameters
    print(f"\nParameters for Localization:")
    pprint.pprint(params); print()

    # Save to YAML file
    io.save_parameters_yaml(params, output_path)


@click.command('localize', help="Localize particles in tomograms using multiprocessing")
# Output Arguments
@click.option('--pick-user-id', type=str, default='octopi',
              help="User ID for the particle picks")
@click.option('--pick-session-id', type=str, default='1',
              help="Session ID for the particle picks")
# Localize Arguments
@click.option('--n-procs', type=int, default=8,
              help="Number of CPU processes to parallelize runs across. Defaults to the max number of cores available or available runs")
@click.option('--pick-objects', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="Specific Objects to Find Picks for")
@click.option('--filter-size', type=int, default=10,
              help="Filter size for localization")
@click.option('--radius-max-scale', type=float, default=1.0,
              help="Maximum radius scale for particles")
@click.option('--radius-min-scale', type=float, default=0.5,
              help="Minimum radius scale for particles")
# Input Arguments
@click.option('--runIDs', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of runIDs to run inference on, e.g., run1,run2,run3 or [run1,run2,run3]")
@click.option('-vs', '--voxel-size', type=float, default=10,
              help="Voxel size for localization")
@click.option('--seg-info', type=str, default='predict,octopi,1',
              callback=lambda ctx, param, value: parsers.parse_target(value),
              help='Query for the organelles segmentations (e.g., "name" or "name,user_id,session_id")')
@click.option('-m', '--method', type=click.Choice(['watershed', 'com'], case_sensitive=False), 
              default='watershed',
              help="Localization method to use")
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the CoPick configuration file")
def cli(config, method, seg_info, voxel_size, runIDs,
        radius_min_scale, radius_max_scale, filter_size, pick_objects, n_procs,
        pick_session_id, pick_user_id):
    """
    CLI entry point for localizing particles in tomograms.
    """

    run_localize(config, method, seg_info, voxel_size, runIDs,
        radius_min_scale, radius_max_scale, filter_size, pick_objects, n_procs,
        pick_session_id, pick_user_id)
    

def run_localize(config, method, seg_info, voxel_size, runIDs,
        radius_min_scale, radius_max_scale, filter_size, pick_objects, n_procs,
        pick_session_id, pick_user_id):
    """
    Run the localize command.
    """
    import octopi.utils.io as io
    import multiprocess as mp
    import copick, os
    
    # Save JSON with Parameters
    root = copick.from_file(config)
    overlay_root = io.remove_prefix(root.config.overlay_root)
    basepath = os.path.join(overlay_root, 'logs')
    os.makedirs(basepath, exist_ok=True)
    output_path = os.path.join(basepath, f'localize-{pick_user_id}_{pick_session_id}.yaml')
    
    save_parameters(
        seg_info=seg_info,
        config=config,
        voxel_size=voxel_size,
        pick_session_id=pick_session_id,
        pick_user_id=pick_user_id,
        method=method,
        radius_min_scale=radius_min_scale,
        radius_max_scale=radius_max_scale,
        filter_size=filter_size,
        output_path=output_path
    )

    # Set multiprocessing start method
    mp.set_start_method("spawn")
    
    pick_particles(
        copick_config_path=config,
        method=method,
        seg_info=seg_info,
        voxel_size=voxel_size,
        pick_session_id=pick_session_id,
        pick_user_id=pick_user_id,
        radius_min_scale=radius_min_scale,
        radius_max_scale=radius_max_scale,
        filter_size=filter_size,
        runIDs=runIDs,
        pick_objects=pick_objects,
        n_procs=n_procs,
    )


if __name__ == "__main__":
    cli()

