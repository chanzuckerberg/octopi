from octopi import cli_context
import rich_click as click

@click.group("nnunet", context_settings=cli_context, no_args_is_help=True)
def cli():
    """nnUNet integration for CoPick cryo-ET datasets."""
    pass


from octopi.nnunet.prepare import cli as prepare_cli  # noqa: E402
from octopi.nnunet.train import cli as train_cli      # noqa: E402
from octopi.nnunet.predict import cli as predict_cli  # noqa: E402

cli.add_command(prepare_cli, "prepare")
cli.add_command(train_cli, "train")
cli.add_command(predict_cli, "segment")
