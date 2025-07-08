from scipy.spatial.transform import Rotation as R
from copick_utils.io import readers
import scipy.ndimage as ndi
from typing import Tuple
import numpy as np
import math

def process_membrane_bound_extract(run,
                                   voxel_size: float,
                                   picks_info: Tuple[str, str, str],
                                   membrane_info: Tuple[str, str, str],
                                   organelle_info: Tuple[str, str, str],
                                   save_user_id: str,
                                   save_session_id: str,
                                   distance_threshold: float):

    """
    Process membrane-bound particles and extract their coordinates and orientations.
    
    Args:
        run: CoPick run object.
        voxel_size: Voxel size for coordinate scaling.
        segmentation_name: Name of the segmentation object.
        segmentation_user_id: User ID for the segmentation.
        segmentation_session_id: Session ID for the segmentation.
        picks_name: Name of the particle picks object.
        picks_user_id: User ID for the particle picks.
        picks_session_id: Session ID for the particle picks.
        save_user_id: User ID for saving processed picks.
        save_session_id: Session ID for saving close picks.
        distance_threshold: Maximum distance to consider a particle close to the membrane.
        organelle_seg: Whether to compute organelle centers from segmentation.
    """  

    # Increment session ID for the second class
    new_session_id = str(int(save_session_id) + 1)  # Convert to string after increment                                  

    # Need Better Error Handing for Missing Picks
    coordinates = readers.coordinates(
        run, 
        picks_info[0], picks_info[1], picks_info[2],
        voxel_size,
        raise_error=False
    )

    # If No Coordinates are Found, Return
    if coordinates is None:
        print(f'[Warning] RunID: {run.name} - No Coordinates Found for {picks_info[0]}, {picks_info[1]}, {picks_info[2]}')
        return

    nPoints = len(coordinates)

    # Determine which Segmentation to Use for Filtering
    if membrane_info is None:
        # Flag to distinguish between organelle and membrane segmentation
        membranes_provided = False
        seg = readers.segmentation(
            run, 
            voxel_size, 
            organelle_info[0],
            user_id=organelle_info[1], 
            session_id=organelle_info[2],
            raise_error=False)
        # If No Segmentation is Found, Return
        if seg is None: return     
        elif nPoints == 0 or np.unique(seg).max() == 0:
            print(f'[Warning] RunID: {run.name} - Organelle-Seg Unique Values: {np.unique(seg)}, nPoints: {nPoints}')
            return                                            
    else:
        # Read both Organelle and Membrane Segmentations
        membranes_provided = True
        seg = readers.segmentation(
            run, 
            voxel_size, 
            membrane_info[0],
            user_id=membrane_info[1], 
            session_id=membrane_info[2],
            raise_error=False)

        organelle_seg = readers.segmentation(
            run, 
            voxel_size, 
            organelle_info[0],
            user_id=organelle_info[1], 
            session_id=organelle_info[2],
            raise_error=False)
        
        # If No Segmentation is Found, Return
        if seg is None or seg is None: return
        elif nPoints == 0 or np.unique(seg).max() == 0:
            print(f'[Warning] RunID: {run.name} - Organelle-Seg Unique Values: {np.unique(seg)}, nPoints: {nPoints}')
            return
        
        # Tempory Solution to Ensure Labels are the Same:
        seg[seg > 0] += 1

    if nPoints > 0:

        # Step 1: Find Closest Points to Segmentation of Interest
        points, closest_labels = closest_organelle_points(
            organelle_seg, 
            coordinates, 
            max_distance=distance_threshold, 
            return_labels_array=True
        )

        # Identify close and far indices
        close_indices = np.where(closest_labels != -1)[0]
        far_indices = np.where(closest_labels == -1)[0]

        # Initialize orientations array
        orientations = np.zeros([nPoints, 4, 4])
        orientations[:,3,3] = 1 

        # Step 2: Get Organelle Centers (Optional if an organelle segmentation is provided)
        organelle_centers = organelle_points(organelle_seg)

        # Step 3: Get All the Rotation Matrices from Euler Angles Based on Normal Vector
        if len(close_indices) > 0:

            # Get Organelle Centers for Close Points
            close_labels = closest_labels[close_indices]
            close_centers = np.array([organelle_centers[str(int(label))] for label in close_labels])

            # Calculate orientations
            for i, idx in enumerate(close_indices):
                rot = mCalcAngles(coordinates[idx], close_centers[i])
                r = R.from_euler('ZYZ', rot, degrees=True)
                orientations[idx,:3,:3] = r.inv().as_matrix()  

        # Swap z and x coordinates (0 and 2) before scaling Back to Angstroms
        coordinates[:, [0, 2]] = coordinates[:, [2, 0]]
        coordinates = coordinates * voxel_size
        
        # Save the close points in CoPick project
        if len(close_indices) > 0:
            try:
                close_picks = run.new_picks(object_name=picks_info[0], user_id=save_user_id, session_id=save_session_id)
            except:
                close_picks = run.get_picks(object_name=picks_info[0], user_id=save_user_id, session_id=save_session_id)[0]
            close_picks.from_numpy(coordinates[close_indices], orientations[close_indices])

        # Save the far points Coordinates in another CoPick pick
        if len(far_indices) > 0:                       
            try:
                far_picks = run.new_picks(object_name=picks_info[0], user_id=save_user_id, session_id=new_session_id)
            except:
                far_picks = run.get_picks(object_name=picks_info[0], user_id=save_user_id, session_id=new_session_id)[0]

            # Assume We Don't Know The Orientation for Anything Far From Membranes
            empty_orientations =  np.zeros(orientations[far_indices].shape)
            empty_orientations[:,-1,-1] = 1
            far_picks.from_numpy(coordinates[far_indices], empty_orientations)


def organelle_points(mask, xyz_order=False): 

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background (label 0)

    coordinates = {}
    for label in unique_labels:
        center_of_mass = ndi.center_of_mass(mask == label)
        if xyz_order:
            center_of_mass = center_of_mass[::-1]
        coordinates[str(label)] = center_of_mass
        # coordinates[str(label)] = ndimage.center_of_mass(mask == label)
    return coordinates     

def closest_organelle_points(mask, coords, min_distance = 0, max_distance=float('inf'), return_labels_array=False):
    """
    Filter points in `coords` based on their proximity to the lysosome membrane.

    Args:
        mask (numpy.ndarray): 3D segmentation mask with integer labels.
        coords (numpy.ndarray): Array of shape (N, 3) with 3D coordinates.
        min_distance (float): Minimum distance threshold for a point to be considered.
        max_distance (float): Maximum distance threshold for a point to be considered.
        return_labels_array (bool): Whether to return the labels array matching the
                                    original order of coords.

    Returns:
        dict: A dictionary where keys are mask labels and values are lists of points
              (3D coordinates) within the specified distance range.
        numpy.ndarray (optional): Array of shape (N,) with the label for each coordinate,
                                   or -1 if the point is outside the specified range.
                                   Only returned if `return_labels_array=True`.
    """

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background (label 0)

    # Combine all mask points and keep track of their labels
    all_mask_points = []
    all_labels = []
    for label in unique_labels:
        label_points = np.argwhere(mask == label)
        all_mask_points.append(label_points)
        all_labels.extend([label] * len(label_points))

    # Combine all mask points and labels into arrays
    all_mask_points = np.vstack(all_mask_points)
    all_labels = np.array(all_labels)    

    # Initialize a dictionary to store filtered points for each label
    label_to_filtered_points = {label: [] for label in unique_labels}
    label_to_filtered_points['far'] = []  # Initialize 'far' key to store rejected points    

    # Initialize an array to store the closest label or -1 for out-of-range points
    closest_labels = np.full(len(coords), -1, dtype=int)

    # Compute the closest label and filter based on distance
    for i, coord in enumerate(coords):
        distances = np.linalg.norm(all_mask_points - coord, axis=1)
        min_index = np.argmin(distances)
        closest_label = all_labels[min_index]
        min_distance_to_membrane = distances[min_index]

        # Check if the distance is within the allowed range
        if min_distance <= min_distance_to_membrane <= max_distance:
            closest_labels[i] = closest_label
            label_to_filtered_points[closest_label].append(coord)
        else:
            label_to_filtered_points['far'].append(coord)

    # Convert lists to NumPy arrays for easier handling
    for label in label_to_filtered_points:
        label_to_filtered_points[label] = np.array(label_to_filtered_points[label])

    if return_labels_array:
        return label_to_filtered_points, closest_labels
    else:
        # Concatenate all points into a single NumPy array
        concatenated_points = np.vstack([points for points in label_to_filtered_points.values() if points.size > 0])        
        return concatenated_points

# Create Class to Estimate Eulers from Centers of Lysate
def mCalcAngles(mbProtein, membrane_point):

    deltaX = mbProtein[0] - membrane_point[0]
    deltaY = mbProtein[1] - membrane_point[1]
    deltaZ = mbProtein[2] - membrane_point[2]
    #-----------------------------
    # angRotion is in [-180, 180]
    #-----------------------------
    angRot = math.atan(deltaY / (deltaX + 1e-30))
    angRot *= (180 / math.pi)
    if deltaX < 0 and deltaY > 0:
        angRot += 180
    elif deltaX < 0 and deltaY < 0:
        angRot -= 180
    angRot = float("{:.2f}".format(angRot))
    #------------------------
    # angTilt is in [0, 180]
    #------------------------
    rXY = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    angTilt = math.atan(rXY / (deltaZ + 1e-30))
    angTilt *= (180 / math.pi)
    if angTilt < 0:
        angTilt += 180.0
    angTilt = float("{:.2f}".format(angTilt))

    return (angRot, angTilt, 0) 