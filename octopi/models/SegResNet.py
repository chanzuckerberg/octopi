from monai.networks.nets import SegResNetDS


class mySegResNet:
    def __init__(self):
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """
        Creates the SegResNetDS model based on provided parameters.

        Args:
            config (dict): Must contain:
                num_classes    (int)        : Number of output segmentation classes.
                init_filters   (int)        : Output channels for the initial convolution.
                blocks_down    (tuple)      : Number of residual blocks per downsampling stage.
                dsdepth        (int)        : Deep supervision depth (1 = no DS).
                act            (str)        : Activation type, e.g. 'relu'.
                norm           (str)        : Normalization type, e.g. 'batch'.
                blocks_up      (tuple|None) : Number of upsample blocks (None = mirror blocks_down).
                upsample_mode  (str)        : Upsampling method: 'deconv', 'nontrainable', 'pixelshuffle'.
                resolution     (tuple|None) : Input voxel spacing for anisotropic kernel support.
        """
        self.config = config
        self.model = SegResNetDS(
            spatial_dims=3,
            init_filters=config['init_filters'],
            in_channels=1,
            out_channels=config['num_classes'],
            act=config.get('act', 'prelu'),
            norm=config.get('norm', 'instance'),
            blocks_down=config.get('blocks_down', (1, 2, 2, 4)),
            blocks_up=config.get('blocks_up'),
            dsdepth=config.get('dsdepth', 1),
            upsample_mode=config.get('upsample_mode', 'deconv'),
            resolution=config.get('resolution'),
        )
        return self.model

    def bayesian_search(self, trial, num_classes: int):
        """
        Optuna search space for SegResNetDS.

        Model size is driven by init_filters and blocks_down.
        Capped at init_filters=32 to fit comfortably on 32GB GPUs (A6000, etc.).
        Deep supervision depth is searched from 1 (off) to 3.
        """
        init_filters = trial.suggest_categorical("init_filters", [16, 32])
        dsdepth = trial.suggest_int("dsdepth", 1, 3)
        # Optuna's CategoricalDistribution only supports primitive types
        # (None/bool/int/float/str). Tuples round-trip through the trial DB
        # as lists and break the distribution-compatibility check.
        blocks_down_choices = {
            "1-2-2-4": (1, 2, 2, 4),
            "1-2-2-2": (1, 2, 2, 2),
            "1-1-2-2": (1, 1, 2, 2),
        }
        blocks_down_key = trial.suggest_categorical("blocks_down", list(blocks_down_choices.keys()))
        blocks_down = blocks_down_choices[blocks_down_key]
        self.config = {
            'architecture': 'SegResNet',
            'num_classes': num_classes,
            'init_filters': init_filters,
            'blocks_down': blocks_down,
            'dsdepth': dsdepth,
            'act': 'prelu',
            'norm': 'instance',
            'blocks_up': (1, 1, 1),
            'upsample_mode': 'deconv',
            'resolution': None,
        }
        return self.build_model(self.config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized. Call build_model() or bayesian_search() first.")
        return self.config
