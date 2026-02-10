import rich_click as click

def from_dataportal(
    config, 
    datasetID,
    overlay_path,
    source_type,
    target_type,
    input_voxel_size = 10,
    output_voxel_size = None):
    """
    Download and process tomograms from the CZI Dataportal.
    
    Args:
        config (str): Path to the copick configuration file
        datasetID (int): ID of the dataset to download
        overlay_path (str): Path to the overlay file
        source_type (str): Name of the tomogram type in the dataportal
        target_type (str): Name to use for the tomogram locally
        input_voxel_size (float): Original voxel size of the tomograms
        output_voxel_size (float, optional): Desired voxel size for downsampling
    """

    from octopi.processing.downsample import FourierRescale
    from octopi.utils.progress import _progress, print_summary
    from copick_utils.io import writers
    import copick   

    # Either load an existing configuration file or create one from datasetID and overlay_path
    if config is not None:
        root = copick.from_file(config)
    elif datasetID is not None and overlay_path is not None:
        root = copick.from_czcdp_datasets(
            [datasetID], overlay_root=overlay_path, 
            output_path='config.json', overlay_fs_args={'auto_mkdir': True}
        )
    else:
        raise ValueError('Either config or datasetID and overlay_path must be provided')

    # If we want to save the tomograms at a different voxel size, we need to rescale the tomograms
    if output_voxel_size is not None and output_voxel_size > input_voxel_size:
        rescale = FourierRescale(input_voxel_size, output_voxel_size)
    else:
        output_voxel_size = None

    # Print Parameter Summary
    print_summary(
        "Download Tomograms",
        datasetID=datasetID, overlay_path=overlay_path,
        config=config, source_type=source_type, target_type=target_type,
        input_voxel_size=input_voxel_size, output_voxel_size=output_voxel_size,
    )

    # Main Loop
    for run in _progress(root.runs):

        # Check if voxel spacing is available
        vs = run.get_voxel_spacing(input_voxel_size)

        if vs is None:
            print(f'No Voxel-Spacing Available for RunID: {run.name}, Voxel-Size: {input_voxel_size}')
            continue
        
        # Check if base reconstruction method is available
        avail_tomos = vs.get_tomograms(source_type)
        if avail_tomos is None: 
            print(f'No Tomograms Available for RunID: {run.name}, Voxel-Size: {input_voxel_size}, Tomo-Type: {source_type}')
            continue

        # Download the tomogram
        if len(avail_tomos) > 0:
            vol = avail_tomos[0].numpy()

            # If we want to save the tomograms at a different voxel size, we need to rescale the tomograms
            if output_voxel_size is None:
                writers.tomogram(run, vol, input_voxel_size, target_type)
            else:
                vol = rescale.run(vol)
                writers.tomogram(run, vol, output_voxel_size, target_type)
    
    print(f'✅ Download Complete!\nDownloaded {len(root.runs)} runs')


@click.command('download')
# Voxel Settings
@click.option('-ovs', '--output-voxel-size', type=float, default=None,
              help="Desired output voxel size for downsampling (optional)")
@click.option('-ivs', '--input-voxel-size', type=float, default=10,
              help="Original voxel size of the tomograms")
# Tomogram Settings
@click.option('-t', '--target-type', type=str, default='denoised',
              help="Local tomogram type name to save in your Copick project.")
@click.option('-s', '--source-type', type=str, default='wbp-denoised-ctfdeconv',
              help="Name of the tomogram type as labeled on the CryoET Data Portal")
# Input Arguments
@click.option('-o', '--overlay', type=click.Path(), default=None,
              help="Path to the overlay directory (required with datasetID)")
@click.option('-ds', '--datasetID', type=int, default=None,
              help="Dataset ID from CZI Dataportal (alternative to config)")
@click.option('-c', '--config', type=click.Path(exists=True), default=None,
              help="Path to the copick configuration file (alternative to datasetID)")
def cli(config, datasetid, overlay, source_type, target_type,
        input_voxel_size, output_voxel_size):
    """
    Download and (optionally) downsample tomograms from the CryoET-DataPortal.
    
    This command fetches reconstructed tomograms from publicly available datasets and saves them 
    to your local copick project. Downsampling is performed via Fourier cropping to preserve 
    high-frequency information while reducing file size.
    
    Two modes of operation:
    
    \b
    1. Download tomograms and downsample to a new voxel size:
       octopi download -c config.json --input-voxel-size 10 --output-voxel-size 20
    
    \b
    2. Create new project from the CryoET Data Portal:
       octopi download --datasetID 10301 --overlay-path ./my_project --input-voxel-size 10
    
    The downloaded tomograms will be stored in your copick project structure with the specified 
    voxel spacing and tomogram type.
    """
    
    from_dataportal(
        config=config,
        datasetID=datasetid,
        overlay_path=overlay,
        source_type=source_type,
        target_type=target_type,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size
    )


if __name__ == "__main__":
    cli()

