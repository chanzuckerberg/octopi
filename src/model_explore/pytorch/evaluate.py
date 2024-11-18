from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import numpy as np

def remove_repeated_picks(coordinates, distanceThreshold, pixelSize = 1):

    # Calculate the distance matrix for the 3D coordinates
    dist_matrix = distance.cdist(coordinates[:, :3]/pixelSize, coordinates[:, :3]/pixelSize)

    # Create a linkage matrix using single linkage method
    Z = linkage(dist_matrix, method='complete')

    # Form flat clusters with a distance threshold to determine groups
    clusters = fcluster(Z, t=distanceThreshold, criterion='distance')

    # Initialize an array to store the average of each group
    unique_coordinates = np.zeros((max(clusters), coordinates.shape[1]))

    # Calculate the mean for each cluster
    for i in range(1, max(clusters) + 1):
        unique_coordinates[i-1] = np.mean(coordinates[clusters == i], axis=0)

    return unique_coordinates    

def compute_metrics(gt_points, pred_points, threshold):
    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)
    
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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy, 
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
