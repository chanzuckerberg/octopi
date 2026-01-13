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

def _pos_neg_from_bg_ratio(bg_ratio: float):
    """
    bg_ratio in [0,1]
    neg is fixed at 1
    pos increases with bg_ratio
    """
    neg = 1
    pos = 1 + int(round(bg_ratio * (max_pos - 1)))
    return pos, neg

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

    # Determine the cropping strategy based on bg_ratio
    if bg_ratio: # Provide Pure Background Crops
        pos, neg = _pos_neg_from_bg_ratio(bg_ratio)
        crop = RandCropByPosNegLabeld(
            keys=['image', 'label'],
            label_key="label",
            spatial_size=[input_dim[0], input_dim[1], input_dim[2]],     
            pos=pos, neg=neg,
            num_samples=num_samples
        )
    else: # Provide Crops Based on Label Classes
        crop = RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[input_dim[0], input_dim[1], input_dim[2]],     
            num_classes=Nclasses,
            num_samples=num_samples
        )

    return Compose([
        # Geometric augmentations
        crop,
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2], max_k=3),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),        
        # Intensity augmentations
        RandomOrder([
            RandScaleIntensityd(keys="image", prob=0.5, factors=(0.85, 1.15)),
            RandShiftIntensityd(keys="image", prob=0.5, offsets=(-0.15, 0.15)),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.85, 1.15)),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.5),  # Reduced noise std
        ]),
    ]) 

    # Augmentations to Explore in the Future: 
    # Intensity-based augmentations
    # RandHistogramShiftd(keys="image", prob=0.5, num_control_points=(3, 5))
    # RandGaussianSmoothd(keys="image", prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),

    # Geometric Transforms
    # RandAffined(
    #     keys=["image", "label"],
    #     rotate_range=(0.1, 0.1, 0.1),  # Rotation angles (radians) for x, y, z axes
    #     scale_range=(0.1, 0.1, 0.1),   # Scale range for isotropic/anisotropic scaling
    #     prob=0.5,                      # Probability of applying the transform
    #     padding_mode="border"          # Handle out-of-bounds values
    # )    

def get_predict_transforms():
    """
    Returns predict transforms.
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image")
    ])
