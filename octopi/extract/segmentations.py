from typing import List, Optional
from octopi.utils import parsers
import rich_click as click

def run_extract_seg(
    config: str,
    object_name: str,
    seg_uri: str,
    user_id: Optional[str],
    session_id: str,
    run_ids: Optional[List[str]],
):
    from octopi.utils import io
    from copick_utils.io import readers, writers
    import numpy as np
    import copick

    # Parse the source segmentation query (the raw multi-class prediction)
    seg_name, seg_user_id, seg_session_id = parsers.parse_target(seg_uri)
    seg_user_id = seg_user_id or 'octopi'
    seg_session_id = seg_session_id or '1'

    # Load the inference log to recover the voxel size and object -> label mapping
    # used when the raw prediction was generated (written by `octopi segment`).
    seg_config = io.get_config(config, seg_name, 'segment', seg_user_id, seg_session_id)
    voxel_size = seg_config['inputs']['voxel_size']
    labels = seg_config['labels']

    if object_name not in labels:
        raise ValueError(
            f"Object '{object_name}' not found in the labels for segmentation "
            f"'{seg_name}:{seg_user_id}/{seg_session_id}'. Available objects: {sorted(labels)}"
        )
    label = labels[object_name]

    # Default the output user ID to the source segmentation's user ID
    user_id = user_id or seg_user_id

    # Load Copick Project and get run IDs if not provided
    root = copick.from_file(config)
    if run_ids is None:
        run_ids = [run.name for run in root.runs]

    print(
        f"\n🔍 Extracting '{object_name}' (label={label}) from "
        f"'{seg_name}:{seg_user_id}/{seg_session_id}' @ {voxel_size} "
        f"-> '{object_name}:{user_id}/{session_id}'\n"
    )

    for run_id in run_ids:
        run = root.get_run(run_id)
        seg = readers.segmentation(run, voxel_size, seg_name, seg_user_id, seg_session_id, verbose=False)
        if seg is None:
            print(f"  ⚠️  {run_id}: no '{seg_name}' segmentation found at voxel size {voxel_size} — skipping.")
            continue

        out_seg = (seg == label).astype(np.uint8)
        writers.segmentation(run, out_seg, user_id, object_name, session_id, voxel_size, multilabel=False)
        print(f"  ✅ {run_id}: extracted {int(out_seg.sum()):,} voxels")

    print("\n✅ Extraction complete.")


@click.command('seg', no_args_is_help=True)
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the CoPick configuration file")
@click.option('-n', '--name', 'object_name', type=str, required=True,
              help="Object name to extract from the raw multi-class prediction (e.g. 'membranes')")
@click.option('-uri', '--seg-uri', type=str, default='predict:octopi/1',
              help='Source segmentation to extract from: "name", "name:user_id", or '
                   '"name:user_id/session_id". Default "predict:octopi/1".')
@click.option('-uid', '--user-id', type=str, default=None,
              help="User ID for the extracted segmentation (defaults to the source segmentation's user ID)")
@click.option('-sid', '--session-id', type=str, default='1',
              help="Session ID for the extracted segmentation")
@click.option('-runs', '--run-ids', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="List of run IDs to process. Defaults to all runs.")
def cli(config, object_name, seg_uri, user_id, session_id, run_ids):
    """
    Extract a single object from a raw multi-class Octopi prediction and save it as its
    own standalone CoPick segmentation.

    Reads the inference log written by `octopi segment` to recover the voxel size and the
    object's integer label within the raw prediction, then writes a binary mask for just
    that object as a new segmentation.

    \b
    Example:
      octopi extract seg --config config.json --name membranes
    """
    run_extract_seg(config, object_name, seg_uri, user_id, session_id, run_ids)


if __name__ == "__main__":
    cli()
