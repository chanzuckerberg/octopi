from octopi.datasets.loader import LoadCopickd
from monai.transforms import (
    Compose, 
    RandFlipd, 
    Orientationd, 
    RandRotate90d, 
    NormalizeIntensityd,
    EnsureChannelFirstd, 
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    ScaleIntensityRanged,  
    RandomOrder,
    RandAffined,
)

def get_transforms():
    """
    Returns non-random transforms.
    """
    return Compose([
        LoadCopickd(),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

def get_random_transforms( input_dim, num_samples, Nclasses, bg_ratio: float = 0.0):
    """
    Input:
        input_dim: tuple of (nx, ny, nz)
        num_samples: int
        Nclasses: int
        bg_ratio: float

    Returns random transforms.
    
    For data with a missing wedge along the first axis (causing smearing in that direction),
    we avoid rotations that would move this artifact to other axes. We only rotate around
    the first axis (spatial_axes=[1, 2]) and avoid flipping along the first axis.
    """

    # Check to ensure bg_ratio is within valid range
    if not (0.0 <= bg_ratio <= 1.0):
        raise ValueError(f"bg_ratio must be in [0,1], got {bg_ratio}")

    # bg_ratio controls background fraction; foreground classes are sampled equally
    if bg_ratio > 0:
        fg_ratio = (1 - bg_ratio) / (Nclasses - 1)
        ratios = [bg_ratio] + [fg_ratio] * (Nclasses - 1)
    else:
        ratios = None  # equal sampling across all classes
    crop = RandCropByLabelClassesd(
        keys=["image", "label"], label_key="label",
        spatial_size=[input_dim[0], input_dim[1], input_dim[2]],
        num_classes=Nclasses, num_samples=num_samples, ratios=ratios
    )

    # Random Affine Transform
    rot = RandAffined(
        keys=["image", "label"],
        prob=0.3,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
    )

    return Compose([
        # Geometric augmentations
        crop, rot,
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2], max_k=3),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),        
        # Intensity augmentations
        RandomOrder([
            RandScaleIntensityd(keys="image", prob=0.3, factors=(0.85, 1.15)),
            RandShiftIntensityd(keys="image", prob=0.3, offsets=(-0.15, 0.15)),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.85, 1.15)),
            RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.5),
        ]),
    ]) 

    # Augmentations to Explore in the Future: 
    # Intensity-based augmentations
    # RandHistogramShiftd(keys="image", prob=0.5, num_control_points=(3, 5))
    # RandGaussianSmoothd(keys="image", prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),  

def get_predict_transforms():
    """
    Returns predict transforms.
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image")
    ])
