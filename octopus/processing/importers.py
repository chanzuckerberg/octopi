from octopus.processing.downsample import FourierRescale
import copick, argparse, mrcfile, glob, os
from copick_utils.writers import write
from tqdm import tqdm

def from_dataportal(
    config, 
    datasetID,
    overlay_path,
    dataportal_name,
    target_tomo_type,
    input_voxel_size = 10,
    output_voxel_size = None):
    
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
                write.tomogram(run, vol, input_voxel_size, target_tomo_type)
            else:
                vol = rescale.run(vol)
                write.tomogram(run, vol, output_voxel_size, target_tomo_type)
    
    print(f'Downloading Complete!! Downloaded {len(root.runs)} runs')

def cli_dataportal():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=False, default=None, help='Path to the config file')
    parser.add_argument('--datasetID', type=int, required=False, default=None, help='Dataset ID')
    parser.add_argument('--overlay-path', type=str, required=False, default=None, help='Path to the overlay file')
    parser.add_argument('--dataportal-name', type=str, required=False, default='wbp', help='Dataportal name')
    parser.add_argument('--target-tomo-type', type=str, required=False, default='wbp', help='Local name')
    parser.add_argument('--input-voxel-size', type=float, required=False, default=10, help='Voxel size')
    parser.add_argument('--output-voxel-size', type=float, required=False, default=None, help='Save voxel size')
    args = parser.parse_args()
    from_dataportal(args.config, args.datasetID, args.overlay_path, args.dataportal_name, args.target_tomo_type, args.input_voxel_size, args.output_voxel_size)

def from_mrcs(
    mrcs_path,
    config,
    target_tomo_type,
    input_voxel_size,
    output_voxel_size = 10):

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
        write.tomogram(run, vol, voxel_size_to_write, target_tomo_type)
    print(f"Processed {len(mrc_files)} files from {mrcs_dir}")


def cli_mrcs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mrcs-path', type=str, required=True, help='Path to the mrcs file')
    parser.add_argument('--config', type=str, required=False, default=None, help='Path to the config file to write tomograms to')
    parser.add_argument('--target-tomo-type', type=str, required=True, help='Reconstruction algorithm used to create the tomogram')
    parser.add_argument('--input-voxel-size', type=float, required=True, help='Voxel size of the tomogram')
    parser.add_argument('--output-voxel-size', type=float, required=False, default=10, help='Save voxel size')
    args = parser.parse_args()
    from_mrcs(args.mrcs_path, args.config, args.target_tomo_type, args.input_voxel_size, args.output_voxel_size)
    
    
