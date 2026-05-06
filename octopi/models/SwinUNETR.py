import monai
from monai.networks.nets import SwinUNETR

_MIN_MONAI_VERSION = "1.5.0"


def _check_monai_version():
    if tuple(int(x) for x in monai.__version__.split(".")[:3]) < tuple(
        int(x) for x in _MIN_MONAI_VERSION.split(".")
    ):
        raise RuntimeError(
            f"MONAI >= {_MIN_MONAI_VERSION} is required (found {monai.__version__}). "
            "Please upgrade: pip install --upgrade monai"
        )


class mySwinUNETR:
    def __init__(self):
        _check_monai_version()
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """
        Create a SwinUNETR model from a config dict.

        Input spatial dimensions must be divisible by 32 (patch_size**5 with
        default patch_size=2).  This is validated automatically during forward().

        Args:
            config (dict): Must contain:
                num_classes  (int)   : Number of output segmentation classes.
                feature_size (int)   : Base embedding dimension. Must be divisible by 12.
                                       Typical values: 24, 36, 48.
                depths       (tuple) : Swin Transformer blocks per stage, e.g. (2, 2, 2, 2).
                num_heads    (tuple) : Attention heads per stage, e.g. (3, 6, 12, 24).
                window_size  (int)   : Local attention window size. Default: 7.
                drop_rate          (float) : Dropout probability.
                attn_drop_rate     (float) : Attention dropout probability.
                dropout_path_rate  (float) : Stochastic depth rate.
                use_checkpoint     (bool)  : Gradient checkpointing (saves memory).
        """
        self.config = config
        self.model = SwinUNETR(
            in_channels=1,
            out_channels=config['num_classes'],
            feature_size=config['feature_size'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config.get('window_size', 7),
            drop_rate=config['drop_rate'],
            attn_drop_rate=config['attn_drop_rate'],
            dropout_path_rate=config['dropout_path_rate'],
            use_checkpoint=config['use_checkpoint'],
            spatial_dims=3,
        )
        return self.model

    def bayesian_search(self, trial, num_classes: int):
        """
        Optuna search space for SwinUNETR, spanning tiny (~5M) to medium (~50M) models.

        Model size is primarily driven by feature_size and depths:
          Tiny   : feature_size=12,  depths=(2,2,2,2)
          Small  : feature_size=24,  depths=(2,2,4,2)
          Medium : feature_size=36,  depths=(2,4,4,2)

        Capped at feature_size=36 to fit comfortably on 32GB GPUs (A6000, etc.)
        with gradient checkpointing enabled.

        (feature_size, num_heads) are searched jointly to guarantee that every
        stage has at least 4 channels per attention head:
          feature_size * 2^stage / num_heads[stage] >= 4 at all stages.

        use_checkpoint is fixed True for search to avoid OOM across trials.
        """
        # Optuna's CategoricalDistribution only supports primitive types
        # (None/bool/int/float/str). Nested tuples round-trip through the trial
        # DB as nested lists and break the distribution-compatibility check.
        fs_nh_choices = {
            "fs12_h3-6-12-24":  (12, (3,  6,  12, 24)),   # tiny:   4 ch/head at all stages
            "fs24_h3-6-12-24":  (24, (3,  6,  12, 24)),   # small:  8 ch/head
            "fs24_h6-12-24-48": (24, (6,  12, 24, 48)),   # small:  4 ch/head, finer attention
            "fs36_h3-6-12-24":  (36, (3,  6,  12, 24)),   # medium: 12 ch/head
            "fs36_h6-12-24-48": (36, (6,  12, 24, 48)),   # medium: 6 ch/head
        }
        fs_nh_key = trial.suggest_categorical("feature_size_num_heads", list(fs_nh_choices.keys()))
        feature_size, num_heads = fs_nh_choices[fs_nh_key]

        depth_choices = {
            "2-2-2-2": (2, 2, 2, 2),   # tiny
            "2-2-4-2": (2, 2, 4, 2),   # small
            "2-4-2-2": (2, 4, 2, 2),   # small (deeper early stages)
            "2-4-4-2": (2, 4, 4, 2),   # medium
            "2-4-4-4": (2, 4, 4, 4),   # medium-large
        }
        depths_key = trial.suggest_categorical("depths", list(depth_choices.keys()))
        depths = depth_choices[depths_key]

        window_size       = trial.suggest_categorical("window_size", [5, 7, 8])
        drop_rate         = trial.suggest_float("drop_rate", 0.0, 0.3)
        attn_drop_rate    = trial.suggest_float("attn_drop_rate", 0.0, 0.3)
        dropout_path_rate = trial.suggest_float("dropout_path_rate", 0.0, 0.2)

        self.config = {
            'architecture': 'SwinUNETR',
            'num_classes': num_classes,
            'feature_size': feature_size,
            'depths': depths,
            'num_heads': num_heads,
            'window_size': window_size,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'dropout_path_rate': dropout_path_rate,
            'use_checkpoint': True,
        }
        return self.build_model(self.config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized. Call build_model() or bayesian_search() first.")
        return self.config
