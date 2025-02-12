from copick_utils.writers import write
import copick, argparse
from tqdm import tqdm

def from_dataportal(
    config, 
    method = 'wbp', 
    processing = 'raw',
    voxel_size = 10):
    
    root = copick.from_file(config)

    # Create a directory for the tomograms
    for run in tqdm(root.runs):

        # Check if voxel spacing is available
        vs = run.get_voxel_spacing(voxel_size)

        if vs is None:
            print(f'No Voxel-Spacing Available for RunID: {run.name}, Voxel-Size: {voxel_size}')
            continue
        
        # Check if base reconstruction method is available
        avail_tomos = vs.get_tomograms(method)
        if avail_tomos is None: 
            print('No Tomograms Available for RunID: {runID}, Voxel-Size: {voxel_size}, Tomo-Type: {method}')
            continue

        # Download the tomogram
        if processing == 'ctfdeconvolved':
            # vol = avail_tomos[0].meta.portal_metadata.portal_metadata.processing
            vol = [obj for obj in avail_tomos if obj.meta.portal_metadata.portal_metadata.processing == 'filtered']
        elif processing == 'raw': # Assume we're looking for pure wbp tomograms
            # vol = avail_tomos[1]
            vol = [obj for obj in avail_tomos if obj.meta.portal_metadata.portal_metadata.processing == 'raw']
            processing = 'wbp'
        else:
            vol = [obj for obj in avail_tomos if obj.meta.portal_metadata.portal_metadata.processing_software == processing]

        if len(vol) == 0:
            print(f'No Tomograms Available for RunID: {run.name}, Voxel-Size: {voxel_size}, Tomo-Type: {method}, Processing: {processing}')
            continue
        else:
            vol = vol[0].numpy()

        # Write the tomogram data
        try: 
            tomogram = vs.new_tomogram(tomo_type=processing)
            tomogram.from_numpy(vol)
        except Exception as e:
            # How can I write a wbp tomogram?
            print(f'Error writing tomogram: {e}')
            tomogram = vs.get_tomograms(processing)[0]
            tomogram.from_numpy(vol)

def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--method', type=str, required=False, default='wbp', help='Tomogram method')
    parser.add_argument('--processing', type=str, required=False, default='raw', help='Tomogram processing')
    parser.add_argument('--voxel-size', type=float, required=False, default=10, help='Voxel size')
    args = parser.parse_args()
    from_dataportal(args.config, args.method, args.processing, args.voxel_size)