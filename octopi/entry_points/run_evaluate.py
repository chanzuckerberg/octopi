import octopi.processing.evaluate as evaluate
from octopi.utils import parsers
from typing import List
import argparse

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

    eval = evaluate.evaluator(
        copick_config_path,
        ground_truth_user_id,
        ground_truth_session_id,
        predict_user_id,
        predict_session_id, 
        object_names=object_names
    )

    eval.run(save_path=save_path, distance_threshold_scale=distance_threshold_scale, runIDs=runIDs)

# Entry point with argparse
def cli():
    """
    CLI entry point for running evaluation.
    """

    parser = argparse.ArgumentParser(
        description='Run evaluation on pick and place predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True, help='Path to the copick configuration file')
    parser.add_argument('--ground-truth-user-id', type=str, required=True, help='User ID for ground truth data')
    parser.add_argument('--ground-truth-session-id', type=str, required=False, default= None, help='Session ID for ground truth data')
    parser.add_argument('--predict-user-id', type=str, required=True, help='User ID for prediction data')
    parser.add_argument('--predict-session-id', type=str, required=False, default= None, help='Session ID for prediction data')
    parser.add_argument('--save-path', type=str, required=False, default= None, help='Path to save evaluation results')
    parser.add_argument('--distance-threshold-scale', type=float, required=False, default = 0.8, help='Compute Distance Threshold Based on Particle Radius')
    parser.add_argument('--object-names', type=parsers.parse_list, default=None, required=False, help='Optional list of object names to evaluate, e.g., ribosome,apoferritin or [ribosome,apoferritin].')
    parser.add_argument('--run-ids', type=parsers.parse_list, default=None, required=False, help='Optional list of run IDs to evaluate, e.g., run1,run2,run3 or [run1,run2,run3].')

    args = parser.parse_args()

    # Call the evaluate function with parsed arguments
    my_evaluator(
        copick_config_path=args.config,
        ground_truth_user_id=args.ground_truth_user_id,
        ground_truth_session_id=args.ground_truth_session_id,
        predict_user_id=args.predict_user_id,
        predict_session_id=args.predict_session_id,
        save_path=args.save_path,
        distance_threshold_scale=args.distance_threshold_scale,
        object_names=args.object_names,
        runIDs=args.run_ids
    )

if __name__ == "__main__":
    cli()