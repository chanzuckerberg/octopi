from octopi.extract.segmentations import cli as extract_seg_cli
from octopi.entry_points.run_extract_mb_picks import cli as mb_picks_cli
import rich_click as click

@click.group('extract', no_args_is_help=True)
def cli():
    """Extract objects or picks from existing pipeline outputs.

    seg: isolate a single object's mask from a multi-class `segment` prediction.
    mb-picks: split existing picks by proximity to a membrane/organelle segmentation.
    """
    pass

cli.add_command(extract_seg_cli)
cli.add_command(mb_picks_cli)

if __name__ == "__main__":
    cli()