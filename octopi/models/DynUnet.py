from monai.networks.nets import DynUNet


class myDynUNet:
    def __init__(self):
        self.model = None
        self.config = None

    def build_model(self, config: dict):
        """
        Creates the DynUNet model based on provided parameters.

        DynUNet requires explicit kernel_size, strides, and upsample_kernel_size
        for each encoder/decoder stage.  The number of stages is determined by
        len(strides).

        Args:
            config (dict): Must contain:
                num_classes         (int)      : Number of output segmentation classes.
                kernel_size         (list)     : Conv kernel sizes per stage, e.g. [3,3,3,3,3].
                strides             (list)     : Conv strides per stage, e.g. [1,2,2,2,2].
                upsample_kernel_size(list)     : Transposed-conv kernel sizes for upsampling.
                filters             (list|None): Output channels per stage. None = auto.
                dropout             (float|None): Dropout probability.
                norm_name           (str|tuple): Normalization type, e.g. 'instance'.
                act_name            (str|tuple): Activation type, e.g. 'leakyrelu'.
                deep_supervision    (bool)     : Enable deep supervision heads.
                deep_supr_num       (int)      : Number of intermediate DS outputs.
                res_block           (bool)     : Use residual conv blocks.
        """
        self.config = config
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=config['num_classes'],
            kernel_size=config['kernel_size'],
            strides=config['strides'],
            upsample_kernel_size=config['upsample_kernel_size'],
            filters=config.get('filters'),
            dropout=config.get('dropout'),
            norm_name=config.get('norm_name', ("INSTANCE", {"affine": True})),
            act_name=config.get('act_name', ("leakyrelu", {"inplace": True, "negative_slope": 0.01})),
            deep_supervision=config.get('deep_supervision', False),
            deep_supr_num=config.get('deep_supr_num', 1),
            res_block=config.get('res_block', True),
        )
        return self.model

    def bayesian_search(self, trial, num_classes: int):
        """
        Optuna search space for DynUNet.

        The architecture is defined by kernel_size and strides per stage.
        Deep supervision is searched over 0-2 intermediate outputs.
        """
        # Network depth / topology — designed for cryo-ET patch sizes (64-160³).
        # First stride=1 is a feature-extraction stage (no downsampling).
        # Remaining strides control downsampling depth.  A trailing stride=1
        # prevents over-downsampling at the bottleneck (same convention as
        # the octopi UNet strides=[2,2,1]).
        #
        # Bottleneck sizes assume 96³ input:
        #   3stage_s22  : 96 → 96 → 48 → 24           (24³ bottleneck)
        #   4stage_s222 : 96 → 96 → 48 → 24 → 12      (12³ bottleneck)
        #   4stage_s221 : 96 → 96 → 48 → 24 → 24      (24³ bottleneck)
        topology = trial.suggest_categorical("topology", [
            "3stage_s22",
            "4stage_s222",
            "4stage_s221",
        ])

        topologies = {
            "3stage_s22": {
                "kernel_size": [3, 3, 3],
                "strides": [1, 2, 2],
                "upsample_kernel_size": [2, 2],
            },
            "4stage_s222": {
                "kernel_size": [3, 3, 3, 3],
                "strides": [1, 2, 2, 2],
                "upsample_kernel_size": [2, 2, 2],
            },
            "4stage_s221": {
                "kernel_size": [3, 3, 3, 3],
                "strides": [1, 2, 2, 1],
                "upsample_kernel_size": [2, 2, 1],
            },
        }
        topo = topologies[topology]

        # MONAI requires deep_supr_num < number of upsample layers, so the
        # upper bound depends on the chosen topology (3stage has 2 upsamples → max 1;
        # 4stage variants have 3 upsamples → max 2).
        max_deep_supr_num = len(topo['upsample_kernel_size']) - 1
        deep_supr_num = trial.suggest_int("deep_supr_num", 0, max_deep_supr_num)
        deep_supervision = deep_supr_num > 0
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        self.config = {
            'architecture': 'DynUNet',
            'num_classes': num_classes,
            'kernel_size': topo['kernel_size'],
            'strides': topo['strides'],
            'upsample_kernel_size': topo['upsample_kernel_size'],
            'filters': None,
            'dropout': dropout,
            'norm_name': ("INSTANCE", {"affine": True}),
            'act_name': ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            'deep_supervision': deep_supervision,
            'deep_supr_num': deep_supr_num,
            'res_block': True,
        }
        return self.build_model(self.config)

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized. Call build_model() or bayesian_search() first.")
        return self.config
