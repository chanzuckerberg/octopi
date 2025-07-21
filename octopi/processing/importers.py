from octopi.processing.downsample import FourierRescale
import copick, argparse, mrcfile, glob, os
from octopi.entry_points import common
from copick_utils.io import writers
from tqdm import tqdm

def from_dataportal(
    config, 
    datasetID,
    overlay_path,
    dataportal_name,
    target_tomo_type,
    input_voxel_size = 10,
    output_voxel_size = None):
    """
    Download and process tomograms from the CZI Dataportal.
    
    Args:
        config (str): Path to the copick configuration file
        datasetID (int): ID of the dataset to download
        overlay_path (str): Path to the overlay file
        dataportal_name (str): Name of the tomogram type in the dataportal
        target_tomo_alg (str): Name to use for the tomogram locally
        input_voxel_size (float): Original voxel size of the tomograms
        output_voxel_size (float, optional): Desired voxel size for downsampling
    """
    if config is not None:
        root = copick.from_file(config)
    elif datasetID is not None and overlay_path is not None:
        root = copick.from_czcdp_datasets([datasetID], overlay_root=overlay_path)
    else:
        raise ValueError('Either config or datasetID and overlay_path must be provided')

    # If we want to save the tomograms at a different voxel size, we need to rescale the tomograms
    if output_voxel_size is not None and output_voxel_size > input_voxel_size:
        rescale = FourierRescale(input_voxel_size, output_voxel_size)

    # Create a directory for the tomograms
    for run in tqdm(root.runs):

        # Check if voxel spacing is available
        vs = run.get_voxel_spacing(input_voxel_size)

        if vs is None:
            print(f'No Voxel-Spacing Available for RunID: {run.name}, Voxel-Size: {input_voxel_size}')
            continue
        
        # Check if base reconstruction method is available
        avail_tomos = vs.get_tomograms(dataportal_name)
        if avail_tomos is None: 
            print(f'No Tomograms Available for RunID: {run.name}, Voxel-Size: {input_voxel_size}, Tomo-Type: {dataportal_name}')
            continue

        # Download the tomogram
        if len(avail_tomos) > 0:
            vol = avail_tomos[0].numpy()

            # If we want to save the tomograms at a different voxel size, we need to rescale the tomograms
            if output_voxel_size is None:
                writers.tomogram(run, vol, input_voxel_size, target_tomo_type)
            else:
                vol = rescale.run(vol)
                writers.tomogram(run, vol, output_voxel_size, target_tomo_type)
    
    print(f'Downloading Complete!! Downloaded {len(root.runs)} runs')

def cli_dataportal_parser(parser_description, add_slurm: bool = False):
    """
    Create argument parser for the dataportal download command.
    
    Args:
        parser_description (str): Description of the parser
        add_slurm (bool): Whether to add SLURM-specific arguments
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, required=False, default=None, help='Path to the config file')
    parser.add_argument('--datasetID', type=int, required=False, default=None, help='Dataset ID')
    parser.add_argument('--overlay-path', type=str, required=False, default=None, help='Path to the overlay file')
    parser.add_argument('--dataportal-name', type=str, required=False, default='wbp', help='Dataportal name')
    parser.add_argument('--target-tomo-type', type=str, required=False, default='wbp', help='Local name')
    parser.add_argument('--input-voxel-size', type=float, required=False, default=10, help='Voxel size')
    parser.add_argument('--output-voxel-size', type=float, required=False, default=None, help='Save voxel size')
    
    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'dataportal-importer', gpus = 0)

    args = parser.parse_args()
    return args

def cli_dataportal():
    """
    Command-line interface for downloading tomograms from the Dataportal.
    Handles argument parsing and calls from_dataportal with the parsed arguments.
    """
    parser_description = "Import tomograms from the Dataportal with optional downsampling with Fourier Cropping"
    args = cli_dataportal_parser(parser_description)
    from_dataportal(args.config, args.datasetID, args.overlay_path, args.dataportal_name, args.target_tomo_type, args.input_voxel_size, args.output_voxel_size)


def from_mrcs(
    mrcs_path,
    config,
    target_tomo_type,
    input_voxel_size,
    output_voxel_size = None):
    """
    Import and process tomograms from local MRC/MRCS files.
    
    Args:
        mrcs_path (str): Path to directory containing MRC/MRCS files
        config (str): Path to the copick configuration file
        target_tomo_type (str): Name to use for the tomogram locally
        input_voxel_size (float): Original voxel size of the tomograms
        output_voxel_size (float, optional): Desired voxel size for downsampling
    """
    # Load Copick Project
    if os.path.exists(config):
        root = copick.from_file(config)
    else:
        raise ValueError('Config file not found')

    # List all .mrc and .mrcs files in the directory
    mrc_files = glob.glob(os.path.join(mrcs_path, "*.mrc")) + glob.glob(os.path.join(mrcs_path, "*.mrcs"))
    if not mrc_files:
        print(f"No .mrc or .mrcs files found in {mrcs_path}")
        return

    # Prepare rescaler if needed
    rescale = None
    if output_voxel_size is not None and output_voxel_size > input_voxel_size:
        rescale = FourierRescale(input_voxel_size, output_voxel_size)        

    # Check if the mrcs file exists
    if not os.path.exists(mrcs_path):
        raise FileNotFoundError(f'MRCs file not found: {mrcs_path}')
    
    for mrc_path in tqdm(mrc_files):

        # Get or Create Run
        runID = os.path.splitext(os.path.basename(mrc_path))[0]
        try:
            run = root.new_run(runID)
        except Exception as e:
            run = root.get_run(runID)

        # Load the mrcs file
        with mrcfile.open(mrc_path) as mrc:
            vol = mrc.data
            # Check voxel size in MRC header vs user input
            mrc_voxel_size = float(mrc.voxel_size.x)  # assuming cubic voxels
            if abs(mrc_voxel_size - input_voxel_size) > 1e-1:
                print(f"WARNING: Voxel size in {mrc_path} header ({mrc_voxel_size}) "
                      f"differs from user input ({input_voxel_size})")

        # Rescale if needed
        if rescale is not None:
            vol = rescale.run(vol)
            voxel_size_to_write = output_voxel_size
        else:
            voxel_size_to_write = input_voxel_size

        # Write the tomogram
        writers.tomogram(run, vol, voxel_size_to_write, target_tomo_type)
    print(f"Processed {len(mrc_files)} files from {mrcs_path}")


def cli_mrcs_parser(parser_description, add_slurm: bool = False):
    """
    Create argument parser for the MRC import command.
    
    Args:
        parser_description (str): Description of the parser
        add_slurm (bool): Whether to add SLURM-specific arguments
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=parser_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input Arguments
    parser.add_argument('--mrcs-path', type=str, required=True, help='Path to the mrcs file')
    parser.add_argument('--config', type=str, required=False, default=None, help='Path to the config file to write tomograms to')
    parser.add_argument('--target-tomo-type', type=str, required=True, help='Reconstruction algorithm used to create the tomogram')
    parser.add_argument('--input-voxel-size', type=float, required=False, default=10, help='Voxel size of the MRC tomogram')
    parser.add_argument('--output-voxel-size', type=float, required=False, default=None, help='Output voxel size (if desired to downsample to lower resolution)')
    
    if add_slurm:
        slurm_group = parser.add_argument_group("SLURM Arguments")
        common.add_slurm_parameters(slurm_group, 'mrcs-importer', gpus = 0)

    args = parser.parse_args()

    return args
    
def cli_mrcs():
    """
    Command-line interface for importing MRC/MRCS files.
    Handles argument parsing and calls from_mrcs with the parsed arguments.
    """
    parser_description = "Import MRC volumes from a directory."
    args = cli_mrcs_parser(parser_description)
    from_mrcs(args.mrcs_path, args.config, args.target_tomo_type, args.input_voxel_size, args.output_voxel_size)
