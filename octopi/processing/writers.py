# This code is adapted from the copick-utils project,
# originally available at: https://github.com/copick/copick-utils/blob/main/src/copick_utils/writers/write.py
# Licensed under the MIT License.

# Copyright (c) 2023 The copick-utils authors

from typing import Any, Dict, List
import numpy as np

def tomogram(
    run,
    input_volume,
    voxel_size=10,
    algorithm="wbp"
):
    """
    Writes a volumetric tomogram into an OME-Zarr format within a Copick directory.

    Parameters:
    -----------
    run : copick.Run
        The current Copick run object.
    input_volume : np.ndarray
        The volumetric tomogram data to be written.
    voxel_size : float, optional
        The size of the voxels in physical units. Default is 10.
    algorithm : str, optional
        The tomographic reconstruction algorithm to use. Default is 'wbp'.

    Returns:
    --------
    copick.Tomogram
        The created or modified tomogram object.
    """
    
    # Retrieve or create voxel spacing
    voxel_spacing = run.get_voxel_spacing(voxel_size)
    if voxel_spacing is None:
        voxel_spacing = run.new_voxel_spacing(voxel_size=voxel_size)
    
    # Check if We Need to Create a New Tomogram for Given Algorithm
    tomogram = voxel_spacing.get_tomogram(algorithm)
    if tomogram is None:
        tomogram = voxel_spacing.new_tomogram(tomo_type=algorithm)
    
    # Write the tomogram data
    tomogram.from_numpy(input_volume)


def segmentation(
    run,
    segmentation_volume,
    user_id,
    name="segmentation",
    session_id="0",
    voxel_size=10,
    multilabel=True
):
    """
    Writes a segmentation into an OME-Zarr format within a Copick directory.

    Parameters:
    -----------
    run : copick.Run
        The current Copick run object.
    segmentation_volume : np.ndarray
        The segmentation data to be written.
    user_id : str
        The ID of the user creating the segmentation.
    name : str, optional
        The name of the segmentation dataset to be created or modified. Default is 'segmentation'.
    session_id : str, optional
        The session ID for this segmentation. Default is '0'.
    voxel_size : float, optional
        The size of the voxels in physical units. Default is 10.
    multilabel : bool, optional
        Whether the segmentation is a multilabel segmentation. Default is True.

    Returns:
    --------
    copick.Segmentation
        The created or modified segmentation object.
    """
    
    # Retrieve or create a segmentation
    segmentations = run.get_segmentations(name=name, user_id=user_id, session_id=session_id)

    # If no segmentation exists or no segmentation at the given voxel size, create a new one
    if len(segmentations) == 0 or any(seg.voxel_size != voxel_size for seg in segmentations):
        segmentation = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=session_id,
            is_multilabel=multilabel,
            user_id=user_id
        )
    else:
        # Overwrite the current segmentation at the specified voxel size if it exists
        segmentation = next(seg for seg in segmentations if seg.voxel_size == voxel_size)
    
    # Write the segmentation data
    segmentation.from_numpy(segmentation_volume, dtype=np.uint8)
