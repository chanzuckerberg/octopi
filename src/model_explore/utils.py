import numpy as np
import zarr


def get_tomogram_array(copick_run, voxel_spacing=10, tomo_type="wbp"):
    voxel_spacing_obj = copick_run.get_voxel_spacing(voxel_spacing)
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    image = zarr.open(tomogram.zarr(), mode="r")["0"]
    return image[:]


def get_segmentation_array(
    copick_run, segmentation_name, voxel_spacing=10, is_multilabel=True
):
    # seg_memb = copick_run.get_segmentations(name="membrane")
    seg_memb = copick_run.get_segmentations()
    seg = copick_run.get_segmentations(
        is_multilabel=is_multilabel, name=segmentation_name, voxel_size=voxel_spacing
    )
    if len(seg) == 0:
        raise ValueError("No segmentations found.")

    segmentation = zarr.open(seg[0].zarr().path, mode="r")["0"][:]
    _, array = list(zarr.open(seg_memb[0].zarr()).arrays())[0]
    seg_membrane = np.array(array[:])
    # seg_membrane = zarr.open(seg_memb[0].zarr().path, mode="r")['0'][:]
    segmentation[seg_membrane == 1] = 1
    return segmentation


def stack_patches(data):
    shape = data.shape
    new_shape = (shape[0] * shape[1],) + shape[2:]
    return data.view(new_shape)
