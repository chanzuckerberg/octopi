from octopi.extract import membranebound_extract as extract
from scipy.spatial.transform import Rotation as R
from copick_utils.io import readers
from scipy.spatial import cKDTree
from typing import Tuple
import numpy as np

def process_midpoint_extract(
    run, 
    voxel_size: float, 
    picks_info: Tuple[str, str, str],
    organelle_info: Tuple[str, str, str],
    distance_min: float, distance_max: float, 
    distance_threshold: float,
    save_session_id: str):

    """
    Process coordinates and extract the mid-point between two neighbor coordinates.

    Args:
        run: CoPick run object.
        voxel_size: Voxel size for coordinate scaling.
        picks_info: Tuple of picks name, user_id, and session_id.
        distance_min: Minimum distance for valid nearest neighbors.
        distance_max: Maximum distance for valid nearest neighbors.
        save_user_id: User ID to save the new picks.
        save_session_id: Session ID to save the new picks.
    """

    # Pull Picks that Are used for Midpoint Extraction
    coordinates = readers.coordinates(
        run, 
        picks_info[0], picks_info[1], picks_info[2],
        voxel_size
    )
    nPoints = len(coordinates)
    
    # Create Base Query for Saving Picks
    save_picks_info = list(picks_info)
    save_picks_info[2] = save_session_id

    # Get Organelle Segmentation
    seg = readers.segmentation(
        run, 
        voxel_size, 
        organelle_info[0],
        user_id=organelle_info[1], 
        session_id=organelle_info[2],
        raise_error=False
    )
    # If No Segmentation is Found, Return
    if seg is None: return     
    elif nPoints == 0 or np.unique(seg).max() == 0:
        print(f'[Warning] RunID: {run.name} - Seg Unique Values: {np.unique(seg)}, nPoints: {nPoints}')
        return        

    if nPoints > 0:

        # Step 1: Find Closest Points to Segmentation of Interest
        points, closest_labels = extract.closest_organelle_points(
            seg, 
            coordinates, 
            max_distance=distance_threshold, 
            return_labels_array=True
        )
    
        # Step 2: Find Midpoints of Closest Points
        midpoints, endpoints = find_midpoints_in_range(
            points, 
            distance_min, 
            distance_max
        )
        
        # Only Process and Save if There Are Any Midpoints
        if len(midpoints) > 0:

            # Step 3: Get Organelle Centers (Optional if an organelle segmentation is provided)
            organelle_centers = extract.organelle_points(seg)

            save_picks_info[1] = picks_info[1] + '-midpoint'
            save_oriented_points(
                run, voxel_size, 
                midpoints, 
                organelle_centers,
                save_picks_info
            )

            save_picks_info[1] = picks_info[1] + '-endpoint'
            save_oriented_points(
                run, voxel_size, 
                endpoints, 
                organelle_centers,
                save_picks_info
            )

def find_midpoints_in_range(lysosome_points, min_distance, max_distance):
    """
    Compute the midpoints of all nearest-neighbor pairs within a given distance range.
    
    Args:
        lysosome_points (dict): A dictionary where keys are lysosome labels and values
                                are NumPy arrays of points associated with each label.
        min_distance (float): Minimum distance for valid nearest neighbors.
        max_distance (float): Maximum distance for valid nearest neighbors.
    
    Returns:
        dict: A dictionary where keys are lysosome labels and values are arrays of midpoints
              for pairs within the specified distance range.
        dict: A dictionary where keys are lysosome labels and values are arrays of endpoints
    """
    midpoints = {}
    endpoints = {}
    
    for label, points in lysosome_points.items():
        if len(points) < 2:
            # Skip if fewer than 2 points (no neighbors to compute)
            midpoints[label] = np.array([])
            continue
        
        # Use cKDTree for efficient neighbor queries
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=2)  # k=2 gets the closest neighbor only
        
        valid_pairs = set()  # Use a set to avoid duplicate pairings
        
        for i, (dist, neighbor_idx) in enumerate(zip(distances[:, 1], indices[:, 1])):
            if min_distance <= dist <= max_distance:
                # Ensure the pair is only added once (sorted tuple prevents duplicates)
                pair = tuple(sorted((i, neighbor_idx)))
                valid_pairs.add(pair)
        
        # Calculate midpoints for unique valid pairs
        midpoints[label] = np.array([
            (points[i] + points[j]) / 2 for i, j in valid_pairs
        ])

        # Get Endpoints
        endpoint_pairs = np.array([
            (points[i], points[j]) for i, j in valid_pairs
        ])
        unique_endpoints = np.unique(endpoint_pairs.reshape(-1, 3), axis=0)
        endpoints[label] = unique_endpoints

    # Return EndPoints and Midpoints
    return midpoints, endpoints

# Assuming `test` is the dictionary or list of arrays
def concatenate_all_midpoints(midpoints_dict):
    """
    Concatenate all arrays of midpoints into a single NumPy array.
    
    Args:
        midpoints_dict (dict): Dictionary with lysosome labels as keys and arrays of midpoints as values.
    
    Returns:
        numpy.ndarray: Single concatenated array of all midpoints.
    """
    all_midpoints = [midpoints for midpoints in midpoints_dict.values() if len(midpoints) > 0]
    if all_midpoints:
        concatenated_array = np.vstack(all_midpoints)
    else:
        concatenated_array = np.array([])  # Return an empty array if no midpoints exist
    return concatenated_array



def save_oriented_points(run, voxel_size, points, organelle_centers, picks_info):

    # Step 5: Concatenate All Midpoints
    concatenated_points = concatenate_all_midpoints(points)   
    nPoints = concatenated_points.shape[0]

    # Initialize orientations array
    orientations = np.zeros([nPoints, 4, 4])
    orientations[:,3,3] = 1 

    # Step 4: Get Rotation Matrices from Euler Angles Based on Normal Vector
    idx = 0
    for key, points in points.items():
        if points.size > 0:
            for point in points:
                rot = extract.mCalcAngles(point, organelle_centers[str(key)])
                r = R.from_euler('ZYZ', rot, degrees=True)
                orientations[idx,:3,:3] = r.inv().as_matrix() 
                idx += 1
        
    # Swap z and x coordinates (0 and 2) before scaling Back to Angstroms
    concatenated_points[:, [0, 2]] = concatenated_points[:, [2, 0]]
    concatenated_points = concatenated_points * voxel_size

    # Step 4: Save Midpoints to Copick
    close_picks = run.new_picks(object_name=picks_info[0], user_id=picks_info[1], session_id=picks_info[2])
    close_picks.from_numpy(concatenated_points, orientations)