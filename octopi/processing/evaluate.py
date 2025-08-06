from copick_utils.io import readers
from scipy.spatial import distance
import copick, json, os, yaml   
from typing import List
import numpy as np

class evaluator:

    def __init__(self, 
                 copick_config: str,
                 ground_truth_user_id: str,
                 ground_truth_session_id: str,
                 prediction_user_id: str,
                 predict_session_id: str,
                 voxel_size: float = 10,
                 beta: float = 4,
                 object_names: List[str] = None):
        
        self.root = copick.from_file(copick_config)
        print('Running Evaluation on the Following Copick Project: ', copick_config)

        self.ground_truth_user_id = ground_truth_user_id
        self.ground_truth_session_id = ground_truth_session_id
        self.prediction_user_id = prediction_user_id
        self.predict_session_id = predict_session_id
        self.voxel_size = voxel_size
        self.beta = beta
        print(f'\nGround Truth Query: \nUserID: {ground_truth_user_id}, SessionID: {ground_truth_session_id}')
        print(f'\nSubmitted Picks: \nUserID: {prediction_user_id}, SessionID: {predict_session_id}\n')

        # Save input parameters
        self.input_params = {
            "copick_config": copick_config,
            "ground_truth_user_id": ground_truth_user_id,
            "ground_truth_session_id": ground_truth_session_id,
            "prediction_user_id": prediction_user_id,
            "predict_session_id": predict_session_id,
        }

        # Get objects that can be Picked
        if not object_names:
            print('No object names provided, using all pickable objects')
            self.objects = [(obj.name, obj.radius) for obj in self.root.pickable_objects if obj.is_particle]
        else:
            # Get valid pickable objects with their radii
            valid_objects = {obj.name: obj.radius for obj in self.root.pickable_objects if obj.is_particle}
            
            # Filter and validate provided object names
            invalid_objects = [name for name in object_names if name not in valid_objects]
            if invalid_objects:
                print('WARNING: The following object names are not valid pickable objects:', invalid_objects)
                print('Valid objects are:', list(valid_objects.keys()))
            
            self.objects = [(name, valid_objects[name]) for name in object_names if name in valid_objects]
            
            if not self.objects:
                raise ValueError("None of the provided object names are valid pickable objects")
                
        print('Using the following valid objects:', [name for name, _ in self.objects])

        # Define object-specific weights
        self.weights = {
            "apo-ferritin": 1,
            "beta-amylase": 0,  # Excluded from scoring
            "beta-galactosidase": 2,
            "ribosome": 1,
            "thyroglobulin": 2,
            "virus-like particle": 1,
        }

    def run(self, 
            save_path: str = None,
            distance_threshold_scale: float = 0.8,
            runIDs: List[str] = None):
    
        # Type check for runIDs
        if runIDs is not None and not (isinstance(runIDs, list) and all(isinstance(x, str) for x in runIDs)):
            raise TypeError("runIDs must be a list of strings")

        run_ids = runIDs if runIDs else [run.name for run in self.root.runs]
        print('\nRunning Metrics Evaluation on the Following RunIDs: ', run_ids)

        metrics = {}
        summary_metrics = {name: {'precision': [], 'recall': [], 'f1_score': [], 'fbeta_score': [], 'accuracy': [], 
                                'true_positives': [], 'false_positives': [], 'false_negatives': []} for name, _ in self.objects}
        
        # For storing the aggregated counts per particle type (across all runs)
        aggregated_counts = {name: {'total_tp': 0, 'total_fp': 0, 'total_fn': 0} for name, _ in self.objects}

        for runID in run_ids:
            # Initialize the nested dictionary for this runID
            metrics[runID] = {}
            run = self.root.get_run(runID)

            for name, radius in self.objects:
                
                # Get Ground Truth and Predicted Coordinates
                gt_coordinates = readers.coordinates(
                    run, name, 
                    self.ground_truth_user_id, self.ground_truth_session_id, 
                    self.voxel_size, raise_error=False
                )
                pred_coordinates = readers.coordinates(
                    run, name,
                    self.prediction_user_id, self.predict_session_id, 
                    self.voxel_size, raise_error=False
                )
                
                # If no reference (GT) points, all candidate points are false positives
                if gt_coordinates is None or len(gt_coordinates) == 0:
                    num_pred_points = pred_coordinates.shape[0] if pred_coordinates is not None else 0
                    metrics[runID][name] = {'precision': 0, 'recall': 0, 'fbeta_score': 0, 'true_positives': 0, 'false_positives': num_pred_points, 'false_negatives': 0}
                    
                    # Update aggregated counts
                    aggregated_counts[name]['total_fp'] += num_pred_points
                    
                    continue

                # If no candidate (predicted) points, all reference points are false negatives
                if pred_coordinates is None or len(pred_coordinates) == 0:
                    num_gt_points = gt_coordinates.shape[0] if gt_coordinates is not None else 0
                    metrics[runID][name] = {'precision': 0, 'recall': 0, 'fbeta_score': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': num_gt_points}
                    
                    # Update aggregated counts
                    aggregated_counts[name]['total_fn'] += num_gt_points
                    
                    continue

                # Compute Distance Threshold Based on Particle Radius
                distance_threshold = (radius/self.voxel_size) * distance_threshold_scale
                metrics[runID][name] = self.compute_metrics(gt_coordinates, pred_coordinates, distance_threshold)         

                # Collect metrics for summary statistics
                for key in summary_metrics[name]:
                    summary_metrics[name][key].append(metrics[runID][name][key])
                
                # Update aggregated counts
                aggregated_counts[name]['total_tp'] += metrics[runID][name]['true_positives']
                aggregated_counts[name]['total_fp'] += metrics[runID][name]['false_positives']
                aggregated_counts[name]['total_fn'] += metrics[runID][name]['false_negatives']

        # Create a new dictionary for summarized metrics
        final_summary_metrics = {}

        # Compute average metrics and standard deviations across runs for each object
        for name, _ in self.objects:
            # Initialize the final summary for the object
            final_summary_metrics[name] = {}

            for key in summary_metrics[name]:
                mu_val = float(np.mean(summary_metrics[name][key]))
                std_val = float(np.std(summary_metrics[name][key]))

                # Populate the new dictionary with structured data
                final_summary_metrics[name][key] = {
                    'mean': mu_val,
                    'std': std_val
                }

        print('\nAverage Metrics Summary:')
        self.print_metrics_summary(final_summary_metrics)          
        
        # Compute Final Kaggle Submission Score using reference approach
        aggregate_fbeta = 0.0
        total_weight = 0.0
        
        print('\nCalculating Final F-beta Score using per-particle approach:')
        for name, counts in aggregated_counts.items():
            tp = counts['total_tp']
            fp = counts['total_fp']
            fn = counts['total_fn']
            
            # Calculate precision and recall for this particle type
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F-beta for this particle type
            particle_fbeta = (1 + self.beta**2) * (precision * recall) / \
                            ((self.beta**2 * precision) + recall) if \
                            ((self.beta**2 * precision) + recall) > 0 else 0
            
            # Get the weight for this particle type
            weight = self.weights.get(name, 1)
            
            # Accumulate weighted F-beta score
            aggregate_fbeta += particle_fbeta * weight
            total_weight += weight
            
            print(f"  {name}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.3f}, " + 
                f"Recall={recall:.3f}, F-beta={particle_fbeta:.3f}, Weight={weight}")
        
        # Normalize by total weight
        final_fbeta = aggregate_fbeta / total_weight if total_weight > 0 else 0
        
        print(f'\nFinal Kaggle Submission Score: {final_fbeta:.3f}')   

        # Save average and detailed metrics with parameters included        
        if save_path:
            self.parameters = {
                "distance_threshold_scale": distance_threshold_scale,
                "runIDs": runIDs,
            }    
            
            os.makedirs(save_path, exist_ok=True)
            summary_metrics = { "input": self.input_params, 
                                "final_fbeta_score": final_fbeta,  
                                "aggregated_particle_scores": {    # Optionally add per-particle details
                                    name: {
                                        "tp": counts['total_tp'],
                                        "fp": counts['total_fp'], 
                                        "fn": counts['total_fn'],
                                        "weight": self.weights.get(name, 1)
                                    } for name, counts in aggregated_counts.items()
                                },
                                "summary_metrics": final_summary_metrics, 
                                "parameters": self.parameters,  }

            # Save average metrics to YAML file
            with open(os.path.join(save_path, 'average_metrics.yaml'), 'w') as f:
                yaml.dump(summary_metrics, f, indent=4, default_flow_style=False, sort_keys=False)
            print(f'\nAverage Metrics saved to {os.path.join(save_path, "average_metrics.yaml")}')
            
            detailed_metrics = { "input": self.input_params,  
                                  "metrics": metrics,
                                 "parameters": self.parameters, }
            with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
                json.dump(detailed_metrics, f, indent=4)
            print(f'Metrics saved to {os.path.join(save_path, "metrics.json")}')

    def compute_metrics(self, 
                        gt_points, 
                        pred_points, 
                        threshold):
        
        gt_points = np.array(gt_points)
        pred_points = np.array(pred_points)
        
        # Calculate distances
        if gt_points.shape[0] == 0:
            # No ground truth points: all predictions are false positives
            fp = pred_points.shape[0]
            fn = 0
            tp = 0
        elif pred_points.shape[0] == 0:
            # No predictions: all ground truth points are false negatives
            fp = 0
            fn = gt_points.shape[0]
            tp = 0
        else:    
            # Calculate distances
            dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')

            # Determine matches within the threshold
            tp = np.sum(np.min(dist_matrix, axis=1) < threshold)
            fp = np.sum(np.min(dist_matrix, axis=1) >= threshold)
            fn = np.sum(np.min(dist_matrix, axis=0) >= threshold)
        
        # Precision, Recall, F1 Score
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = tp / (tp + fp + fn)  # Note: TN not considered here

         # Compute F_beta using the formula
        if (self.beta**2 * precision + recall) > 0:
            fbeta = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall)
        else:
            fbeta = 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fbeta_score': fbeta,
            'accuracy': accuracy, 
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def print_metrics_summary(self, metrics_dict):
        for name, metrics in metrics_dict.items():
            recall = metrics['recall']
            precision = metrics['precision']
            f1_score = metrics['f1_score']
            fbeta_score = metrics['fbeta_score']
            false_positives = metrics['false_positives']
            false_negatives = metrics['false_negatives']
            
            # Format the metrics for the current object
            formatted_metrics = (
                f"Recall: {recall['mean']:.3f} ± {recall['std']:.3f}, "
                f"Precision: {precision['mean']:.3f} ± {precision['std']:.3f}, "
                f"F1 Score: {f1_score['mean']:.3f} ± {f1_score['std']:.3f}, "
                f"F_beta Score: {fbeta_score['mean']:.3f} ± {fbeta_score['std']:.3f}, "
                f"False_Positives: {false_positives['mean']:.1f} ± {false_positives['std']:.1f}, "
                f"False_Negatives: {false_negatives['mean']:.1f} ± {false_negatives['std']:.1f}"
            )
            
            # Print the object name and its metrics
            print(f"{name}: [{formatted_metrics}]")
        print()