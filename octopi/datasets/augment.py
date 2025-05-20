from monai.transforms import (
    Compose, 
    RandFlipd, 
    Orientationd, 
    RandRotate90d, 
    NormalizeIntensityd,
    EnsureChannelFirstd, 
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
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

def get_random_transforms( input_dim, num_samples, Nclasses):
    """
    Input:
        input_dim: tuple of (nx, ny, nz)
        num_samples: int
        Nclasses: int

    Returns random transforms.
    
    For data with a missing wedge along the first axis (causing smearing in that direction),
    we avoid rotations that would move this artifact to other axes. We only rotate around
    the first axis (spatial_axes=[1, 2]) and avoid flipping along the first axis.
    """
    return Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[input_dim[0], input_dim[1], input_dim[2]],     
            num_classes=Nclasses,
            num_samples=num_samples
        ),
        # Only rotate around the first axis (keeping the missing wedge orientation consistent)
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2], max_k=3),
        # Avoid flipping along the first axis (where the missing wedge is)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Removed
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),        
        RandomOrder([
            # Intensity augmentations are still appropriate
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
