from octopi.utils import parsers
from octopi import cli_context
import rich_click as click
from typing import List

def my_evaluator(
    copick_config_path: str,
    ground_truth_user_id: str,
    ground_truth_session_id: str,
    predict_user_id: str,
    predict_session_id: str,
    save_path: str,
    distance_threshold_scale: float,
    object_names: List[str] = None,
    runIDs: List[str] = None
    ):
    import octopi.processing.evaluate as evaluate

    eval = evaluate.evaluator(
        copick_config_path,
        ground_truth_user_id,
        ground_truth_session_id,
        predict_user_id,
        predict_session_id, 
        object_names=object_names
    )

    eval.run(save_path=save_path, distance_threshold_scale=distance_threshold_scale, runIDs=runIDs)


@click.command('evaluate', context_settings=cli_context)
# Output Arguments
@click.option('-o','--output', type=click.Path(), default='scores',
              help="Path to save evaluation results")
# Evaluation Parameters
@click.option('-names','--object-names', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="Optional list of object names to evaluate, e.g., ribosome,apoferritin")
@click.option('-dts','--distance-threshold-scale', type=float, default=0.8,
              help="Compute Distance Threshold Based on Particle Radius")
# Input Arguments
@click.option('--run-ids', type=str, default=None,
              callback=lambda ctx, param, value: parsers.parse_list(value) if value else None,
              help="Optional list of run IDs to evaluate, e.g., run1,run2,run3 or [run1,run2,run3]")
@click.option('-psid', '--predict-session-id', type=str,
              default='1', help="Session ID for prediction data")
@click.option('-puid','--predict-user-id', type=str, required=True,
              default='octopi', help="User ID for prediction data")
@click.option('-gtsid','--ground-truth-session-id', type=str, default=None,
              help="Session ID for ground truth data")
@click.option('-gtuid','--ground-truth-user-id', type=str, required=True,
              help="User ID for ground truth data")
@click.option('-c', '--config', type=click.Path(exists=True), required=True,
              help="Path to the copick configuration file")
def cli(config, ground_truth_user_id, ground_truth_session_id,
        predict_user_id, predict_session_id, run_ids,
        distance_threshold_scale, object_names,
        save_path):
    """
    CLI entry point for running evaluation.
    """

    # Call the evaluate function with parsed arguments
    my_evaluator(
        copick_config_path=config,
        ground_truth_user_id=ground_truth_user_id,
        ground_truth_session_id=ground_truth_session_id,
        predict_user_id=predict_user_id,
        predict_session_id=predict_session_id,
        save_path=save_path,
        distance_threshold_scale=distance_threshold_scale,
        object_names=object_names,
        runIDs=run_ids
    )


if __name__ == "__main__":
    cli()