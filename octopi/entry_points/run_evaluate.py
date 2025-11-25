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
        output):
    """
    Evaluate particle localization performance against ground truth annotations.
    
    This command compares predicted particle picks against expert annotations using distance-based 
    matching. A prediction is considered correct (true positive) if it falls within a specified 
    distance threshold of a ground truth annotation. The threshold is defined as a fraction of 
    the particle's radius (default: 0.8 = 80% of radius).
    
    Computed metrics include:
      • Precision: Fraction of predictions that match ground truth
      • Recall: Fraction of ground truth particles that were detected
      • F-beta scores: Harmonic mean of precision and recall (with configurable beta)
    
    Results are saved as a YAML file containing per-object and aggregate statistics, making it 
    easy to track model performance across experiments and compare different localization methods.
    
    \b
    Examples:
      # Evaluate predictions against Data Portal annotations
      octopi evaluate -c config.json \\
        -gtuid data-portal -gtsid 0 \\
        -puid octopi -psid 1 \\
        -o evaluation_results.yaml
    """

    # Call the evaluate function with parsed arguments
    my_evaluator(
        copick_config_path=config,
        ground_truth_user_id=ground_truth_user_id,
        ground_truth_session_id=ground_truth_session_id,
        predict_user_id=predict_user_id,
        predict_session_id=predict_session_id,
        save_path=output,
        distance_threshold_scale=distance_threshold_scale,
        object_names=object_names,
        runIDs=run_ids
    )


if __name__ == "__main__":
    cli()