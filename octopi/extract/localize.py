from skimage.morphology import binary_erosion, binary_dilation, ball
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import watershed
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from typing import List, Optional, Tuple
from skimage.measure import regionprops
from copick_utils.io import readers
from scipy.spatial import distance
from dataclasses import dataclass
import scipy.ndimage as ndi
from tqdm import tqdm
import numpy as np
import gc

def process_localization(run,  
                          objects, 
                          seg_info: Tuple[str, str, str],
                          method: str = 'com', 
                          voxel_size: float = 10,
                          filter_size: int = None,
                          radius_min_scale: float = 0.5, 
                          radius_max_scale: float = 1.0,
                          pick_session_id: str = '1',
                          pick_user_id: str = 'monai'): 

    # Check if method is valid
    if method not in ['watershed', 'com']:
        raise ValueError(f"Invalid method '{method}'. Expected 'watershed' or 'com'.")

    # Get Segmentation with Error Handling
    try:
        seg = readers.segmentation(
            run, float(voxel_size), 
            seg_info[0], 
            user_id=seg_info[1], 
            session_id=seg_info[2],
            raise_error=False)

        # Preprocess Segmentation
        # seg = preprocess_segmentation(seg, voxel_size, objects)

        # If No Segmentation is Found, Return
        if seg is None:
            print(f"No segmentation found for {run.name}.")
            return

    except Exception as e:
        print(f"[ERROR] - Occurred while reading segmentation from {run.name}: {e}")
        return
    
    # Iterate through all user pickable objects
    for obj in objects:

        # Extract Particle Radius from Root
        min_radius = obj[2] * radius_min_scale / voxel_size
        max_radius = obj[2] * radius_max_scale / voxel_size

        if method == 'watershed':
            points = extract_particle_centroids_via_watershed(seg, obj[1], filter_size, min_radius, max_radius)
        elif method == 'com': 
            points = extract_particle_centroids_via_com(seg, obj[1], min_radius, max_radius)
        points = np.array(points)

        # Save Coordinates if any 3D points are provided
        if points.size > 2:

            # Remove Picks that are too close to each other
            points = remove_repeated_picks(points, min_radius)

            # Swap the coordinates to match the expected format
            points = points[:,[2,1,0]] 

            # Convert the Picks back to Angstrom
            points *= voxel_size

            # Save Picks - Overwrite if exists
            picks = run.new_picks(
                object_name = obj[0], session_id = pick_session_id, 
                user_id=pick_user_id, exist_ok=True)

            # Assign Identity As Orientation
            orientations = np.zeros([points.shape[0], 4, 4])
            orientations[:,:3,:3] = np.identity(3)
            orientations[:,3,3] = 1

            picks.from_numpy( points, orientations )
        else:
            print(f"{run.name} didn't have any available picks for {obj[0]}!")


def extract_particle_centroids_via_watershed(
        segmentation, 
        segmentation_idx, 
        maxima_filter_size, 
        min_particle_radius, 
        max_particle_radius):
    """
    Process a specific label in the segmentation, extract centroids, and save them as picks.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array.
        segmentation_idx (int): The specific label from the segmentation to process.
        maxima_filter_size (int): Size of the maximum detection filter.
        min_particle_size (int): Minimum size threshold for particles.
        max_particle_size (int): Maximum size threshold for particles.
    """

    if maxima_filter_size is None or maxima_filter_size <= 0:
        raise ValueError('Enter a Non-Zero Filter Size!')

    # Calculate minimum and maximum particle volumes based on the given radii
    min_particle_size = (4 / 3) * np.pi * (min_particle_radius ** 3) 
    max_particle_size = (4 / 3) * np.pi * (max_particle_radius ** 3)

    # Create a binary mask for the specific segmentation label
    binary_mask = (segmentation == segmentation_idx).astype(np.uint8)

    # Skip if the segmentation label is not present
    if np.sum(binary_mask) == 0:
        print(f"No segmentation with label {segmentation_idx} found.")
        return

    # Structuring element for erosion and dilation
    struct_elem = ball(1)
    eroded = binary_erosion(binary_mask, struct_elem)

    dilated = binary_dilation(eroded, struct_elem)

    # Distance transform and local maxima detection
    distance = ndi.distance_transform_edt(dilated)
    local_max = (distance == ndi.maximum_filter(distance, footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))))

    # Watershed segmentation
    markers, _ = ndi.label(local_max)
    del local_max
    gc.collect()

    watershed_labels = watershed(-distance, markers, mask=dilated)
    distance, markers, dilated = None, None, None
    del distance, markers, dilated
    gc.collect()

    # Extract region properties and filter based on particle size
    all_centroids = []
    for region in regionprops(watershed_labels):
        if min_particle_size <= region.area <= max_particle_size:

            # Option 1: Use all centroids
            all_centroids.append(region.centroid)

    return all_centroids

def extract_particle_centroids_via_com(
        segmentation, 
        segmentation_idx, 
        min_particle_radius, 
        max_particle_radius
    ):
    """
    Process a specific label in the segmentation, extract centroids, and save them as picks.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array.
        segmentation_idx (int): The specific label from the segmentation to process.
        min_particle_size (int): Minimum size threshold for particles.
        max_particle_size (int): Maximum size threshold for particles.
    """

    # Calculate minimum and maximum particle volumes based on the given radii
    min_particle_size = (4 / 3) * np.pi * (min_particle_radius ** 3) 
    max_particle_size = (4 / 3) * np.pi * (max_particle_radius ** 3)

    # Create a binary mask for the specific segmentation label
    label_objs, _ = ndi.label(segmentation == segmentation_idx)

    # Filter Candidates based on Object Size
    # Get the sizes of all objects
    object_sizes = np.bincount(label_objs.flat)

    # Filter the objects based on size
    valid_objects = np.where((object_sizes > min_particle_size) & (object_sizes < max_particle_size))[0]                        

    # Estimate Coordiantes from CoM for LabelMaps
    octopiCoords = []
    for object_num in tqdm(valid_objects):
        com = ndi.center_of_mass(label_objs == object_num)
        swapped_com = (com[2], com[1], com[0])
        octopiCoords.append(swapped_com)
   
    return octopiCoords

def remove_repeated_picks(coordinates: np.ndarray,
                               distance_threshold: float) -> np.ndarray:
    if coordinates is None or len(coordinates) == 0:
        return coordinates
    if len(coordinates) == 1:
        return coordinates.copy()

    pts = coordinates[:, :3]
    tree = cKDTree(pts)
    # Sparse neighbor graph: edges between points within threshold
    pairs = tree.sparse_distance_matrix(tree, distance_threshold, output_type='coo_matrix')
    n = len(coordinates)
    # Make it symmetric and include self-loops
    A = coo_matrix((np.ones_like(pairs.data), (pairs.row, pairs.col)), shape=(n, n))
    A = A.maximum(A.T)  # undirected
    A.setdiag(1)

    n_comp, labels = connected_components(A, directed=False)
    out = np.vstack([coordinates[labels == k].mean(axis=0) for k in range(n_comp)])
    return out

