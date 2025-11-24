from octopi import cli_context
import rich_click as click
from octopi.entry_points import groups  # Import groups to set OPTION_GROUPS
from octopi.entry_points.run_train import cli as train_model
from octopi.entry_points.run_optuna import cli as model_explore
from octopi.entry_points.run_create_targets import cli as create_targets
from octopi.entry_points.run_segment_predict import cli as inference
from octopi.entry_points.run_localize import cli as localize
from octopi.entry_points.run_evaluate import cli as evaluate
from octopi.entry_points.run_extract_mb_picks import cli as mb_extract

@click.group(context_settings=cli_context)
def routines():
    """Octopi 🐙: 🛠️ Tools for Finding Proteins in 🧊 cryo-ET data"""
    pass

routines.add_command(train_model)
routines.add_command(create_targets)
routines.add_command(inference)
routines.add_command(localize)
routines.add_command(model_explore)
routines.add_command(evaluate)
routines.add_command(mb_extract)

@click.group(context_settings=cli_context)
def slurm_routines():
    """Slurm-Octopi 🐙: 🛠️ Tools for Finding Proteins in 🧊 cryo-ET data"""
    pass

