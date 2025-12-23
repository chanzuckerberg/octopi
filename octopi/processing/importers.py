import rich_click as click

def import_tomos(
    config, 
    path,
    tomo_alg,
    ivs = 10,
    ovs = None):
    """
    Import MRC tomograms from a folder into a copick project.
    
    Args:
        config (str): Path to the copick configuration file
        path (str): Path to the folder containing the tomograms
        tomo_alg (str): Local tomogram type name to save in your Copick project
        ivs (float): Original voxel size of the tomograms
        ovs (float): Desired output voxel size for downsampling (optional)
    """
    from octopi.utils.progress import _progress, print_summary
    from octopi.processing.downsample import FourierRescale
    from copick_utils.io import writers
    import copick, os, glob, mrcfile

    # Either load the config file or create a new project
    if os.path.isfile(config):
        root = copick.from_file(config)
    else:
        raise ValueError('Config file does not exist')

    # If we want to save the tomograms at a different voxel size, we need to rescale the tomograms
    if ovs is not None and ovs > ivs:
        rescale = FourierRescale(ivs, ovs)
    else:
        rescale = None

    # Print Parameter Summary
    print_summary(
        "Import Tomograms",
        path=path, tomo_alg=tomo_alg,
        config=config, ivs=ivs, ovs=ovs,
    )
    
    # Get the list of tomograms in the folder
    tomograms = glob.glob(os.path.join(path, '*.mrc'))

    # Check if no tomograms were found
    if len(tomograms) == 0:
        raise ValueError('No tomograms found in the folder')

    # Main Loop
    for tomogram in _progress(tomograms):

        # Read the tomogram and the associated runID
        with mrcfile.open(tomogram) as mrc:
            vol = mrc.data.copy()
            vs = float(mrc.voxel_size.x) # Assuming cubic voxels
        
        # Check if the voxel size in the tomogram matches the provided input voxel size
        if vs != ivs and rescale is not None:
            print('[WARNING] Voxel size in tomogram does not match the provided input voxel size. Using voxel size from tomogram for downsampling.')
            ivs = vs  # Override voxel size if it doesn't match the expected input voxel size
            rescale = FourierRescale(vs, ovs)
        # Assume that if a voxel size is 1, the MRC didnt' have a voxel size set
        elif vs != 1 and vs != ivs: 
            ivs = vs

        # If we want to save the tomograms at a different voxel size, 
        # we need to rescale the tomograms
        if ovs is not None:
            vol = rescale.run(vol)

        # Get the runID from the tomogram name
        runID = tomogram.split('/')[-1].split('.')[0]

        # Get the run from the project, create new run if it doesn't exist
        run = root.get_run(runID)
        if run is None:
            run = root.new_run(runID)

        # Add the tomogram to the project
        writers.tomogram(run, vol, ovs if ovs is not None else ivs, tomo_alg)
    
    print(f'✅ Import Complete! Imported {len(tomograms)} tomograms')


@click.command('import')
# Input Arguments
@click.option('-p', '--path', type=click.Path(exists=True), default=None, required=True,
              help="Path to the folder containing the tomograms")
@click.option('-c', '--config', type=click.Path(exists=True), default=None,
              help="Path to the copick configuration file (alternative to datasetID)")
# Tomogram Settings
@click.option('-alg', '--tomo-alg', type=str, default='denoised',
              help="Local tomogram type name to save in your Copick project.")
# Voxel Settings
@click.option('-ovs', '--output-voxel-size', type=float, default=None,
              help="Desired output voxel size for downsampling (optional)")
@click.option('-ivs', '--input-voxel-size', type=float, default=10,
              help="Original voxel size of the tomograms")
def cli(config, tomo_alg,
        input_voxel_size, output_voxel_size, path):
    """
    Import MRC tomograms from a folder into a copick project.
    
    This command imports MRC tomograms from a folder into a copick project.
    
    Example Usage:
    
    octopi import -c config.json -p /path/to/tomograms -alg denoised -ivs 5 -ovs 10 (downsample to 10Å)
    
    octopi import -c config.json -p /path/to/tomograms -alg denoised -ovs 10 (will read the voxel size from the tomograms)
    """

    print(f'🚀 Starting Tomogram Import...')
    import_tomos(config=config, path=path, tomo_alg=tomo_alg, ivs=input_voxel_size, ovs=output_voxel_size)

