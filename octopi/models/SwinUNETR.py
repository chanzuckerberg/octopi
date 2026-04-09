from monai.networks.nets import SwinUNETR

class mySwinUNETR:
    def __init__(self):
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """
        Create a SwinUNETR model from a config dict.

        SwinUNETR requires img_size at construction time because the Swin Transformer
        encodes positional information relative to a fixed window grid.  img_size is
        taken from config['dim_in'] and must be divisible by 32.

        Args:
            config (dict): Must contain:
                dim_in       (int)   : Isotropic patch size (e.g. 96). Must be divisible by 32.
                num_classes  (int)   : Number of output segmentation classes.
                feature_size (int)   : Base embedding dimension. Must be divisible by 12.
                                       Typical values: 24, 36, 48.
                depths       (tuple) : Swin Transformer blocks per stage, e.g. (2, 2, 2, 2).
                num_heads    (tuple) : Attention heads per stage, e.g. (3, 6, 12, 24).
                drop_rate          (float) : Dropout probability.
                attn_drop_rate     (float) : Attention dropout probability.
                dropout_path_rate  (float) : Stochastic depth rate.
                use_checkpoint     (bool)  : Gradient checkpointing (saves memory).
        """
        self.config = config
        dim_in = config['dim_in']
        self.model = SwinUNETR(
            img_size=(dim_in, dim_in, dim_in),
            in_channels=1,
            out_channels=config['num_classes'],
            feature_size=config['feature_size'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            drop_rate=config['drop_rate'],
            attn_drop_rate=config['attn_drop_rate'],
            dropout_path_rate=config['dropout_path_rate'],
            use_checkpoint=config['use_checkpoint'],
            spatial_dims=3,
        )
        return self.model

    def bayesian_search(self, trial, num_classes: int):
        """
        Optuna search space for SwinUNETR, spanning tiny (~5M) to large (~200M+) models.

        Model size is primarily driven by feature_size and depths:
          Tiny   : feature_size=12,  depths=(2,2,2,2)
          Small  : feature_size=24,  depths=(2,2,4,2)
          Medium : feature_size=36,  depths=(2,4,4,2)
          Large  : feature_size=48,  depths=(4,4,8,4)
          XLarge : feature_size=48,  depths=(4,8,8,8)

        (feature_size, num_heads) are searched jointly to guarantee that every
        stage has at least 4 channels per attention head:
          feature_size * 2^stage / num_heads[stage] >= 4 at all stages.

        use_checkpoint is fixed True for search to avoid OOM across trials.
        """
        dim_in = trial.suggest_categorical("dim_in", [64, 96, 128])

        # Joint (feature_size, num_heads) choices — all guarantee ≥ 4 ch/head
        fs_nh = trial.suggest_categorical("feature_size_num_heads", [
            (12, (3,  6,  12,  24)),   # tiny:   4 ch/head at all stages
            (24, (3,  6,  12,  24)),   # small:  8 ch/head
            (24, (6,  12, 24,  48)),   # small:  4 ch/head, finer attention
            (36, (3,  6,  12,  24)),   # medium: 12 ch/head
            (36, (6,  12, 24,  48)),   # medium: 6 ch/head
            (48, (3,  6,  12,  24)),   # large:  16 ch/head
            (48, (6,  12, 24,  48)),   # large:  8 ch/head
            (48, (12, 24, 48,  96)),   # large:  4 ch/head, finest attention
        ])
        feature_size, num_heads = fs_nh

        depths = trial.suggest_categorical("depths", [
            (2, 2, 2, 2),   # tiny
            (2, 2, 4, 2),   # small
            (2, 4, 2, 2),   # small (deeper early stages)
            (2, 4, 4, 2),   # medium
            (2, 4, 4, 4),   # medium-large
            (4, 4, 4, 4),   # large
            (4, 4, 8, 4),   # large (deep bottleneck)
            (4, 8, 8, 4),   # extra large
            (4, 8, 8, 8),   # extra large (deep decoder)
        ])

        drop_rate         = trial.suggest_float("drop_rate", 0.0, 0.3)
        dropout_path_rate = trial.suggest_float("dropout_path_rate", 0.0, 0.2)

        self.config = {
            'architecture': 'SwinUNETR',
            'num_classes': num_classes,
            'dim_in': dim_in,
            'feature_size': feature_size,
            'depths': depths,
            'num_heads': num_heads,
            'drop_rate': drop_rate,
            'attn_drop_rate': 0.0,
            'dropout_path_rate': dropout_path_rate,
            'use_checkpoint': True,
        }
        return self.build_model(self.config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized. Call build_model() or bayesian_search() first.")
        return self.config
