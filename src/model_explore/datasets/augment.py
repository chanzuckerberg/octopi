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

def get_random_transforms( crop_size, num_samples, Nclasses):
        """
        Returns random transforms.
        """
        return Compose([
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[crop_size, crop_size, crop_size],
                num_classes=Nclasses,
                num_samples=num_samples
            ),
            RandomOrder([
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandScaleIntensityd(keys="image", factors=(0.9, 1.1), prob=0.5),
                RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
                RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.9, 1.1)),
                RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),
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

def get_validation_transforms( crop_size, num_samples, Nclasses):
        """
        Returns validation transforms.
        """
        return Compose([
            RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[crop_size, crop_size, crop_size],
                    num_classes=Nclasses,
                    num_samples=num_samples, 
            ),
        ])

def get_predict_transforms():
    """
    Returns predict transforms.
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image")
    ])