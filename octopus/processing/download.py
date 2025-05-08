from octopus.processing.downsample import FourierRescale
from copick_utils.writers import write
import copick, argparse
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

def cli():
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