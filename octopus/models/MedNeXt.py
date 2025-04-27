from monai.networks.nets import MedNeXt
import torch.nn as nn
import torch

class myMedNeXt:
    def __init__(self ):
        # Placeholder for the model and config
        self.model = None
        self.config = None

    def build_model(
        self,
        init_filters=32,
        encoder_expansion_ratio=2,
        decoder_expansion_ratio=2,
        bottleneck_expansion_ratio=2,
        kernel_size=7,
        deep_supervision=False,
        use_residual_connection=False,
        norm_type="group",
        global_resp_norm=False,
        blocks_down=(2, 2, 2, 2),
        blocks_bottleneck=2,
        blocks_up=(2, 2, 2, 2),
    ):
        """
        Create the MedNeXt model with the specified hyperparameters.
        Note: For cryoET with small objects, a shallower network might help preserve details.
        """
        self.model = MedNeXt(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.num_classes,
            init_filters=init_filters,
            encoder_expansion_ratio=encoder_expansion_ratio,
            decoder_expansion_ratio=decoder_expansion_ratio,
            bottleneck_expansion_ratio=bottleneck_expansion_ratio,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            use_residual_connection=use_residual_connection,
            norm_type=norm_type,
            global_resp_norm=global_resp_norm,
            blocks_down=blocks_down,
            blocks_bottleneck=blocks_bottleneck,
            blocks_up=blocks_up,
        ).to(self.device)

    def bayesian_search(self, trial):
        """
        Defines the Bayesian optimization search space and builds the model with suggested parameters.
        The search space has been adapted for cryoET applications:
          - Small kernel sizes (3 or 5) to capture fine details.
          - Choice of a shallower vs. deeper architecture to balance resolution and feature extraction.
          - Robust normalization options for low signal-to-noise data.
        """
        # Core hyperparameters
        init_filters = trial.suggest_categorical("init_filters", [16, 32])
        encoder_expansion_ratio = trial.suggest_int("encoder_expansion_ratio", 1, 3)
        decoder_expansion_ratio = trial.suggest_int("decoder_expansion_ratio", 1, 3)
        bottleneck_expansion_ratio = trial.suggest_int("bottleneck_expansion_ratio", 1, 4)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        deep_supervision = trial.suggest_categorical("deep_supervision", [True, False])
        norm_type = trial.suggest_categorical("norm_type", ["group", "instance"])
        # For extremely low SNR, you might opt to disable global response normalization
        global_resp_norm = trial.suggest_categorical("global_resp_norm", [False])
        
        # Architecture: shallow vs. deep.
        # For small objects, a shallower network (fewer downsampling stages) may preserve spatial detail.
        architecture = trial.suggest_categorical("architecture", ["shallow", "deep"])
        if architecture == "shallow":
            blocks_down = (2, 2, 2)         # 3 downsampling stages
            blocks_bottleneck = 2
            blocks_up = (2, 2, 2)
        else:
            blocks_down = (2, 2, 2, 2)        # 4 downsampling stages
            blocks_bottleneck = 2
            blocks_up = (2, 2, 2, 2)

        self.build_model(
            init_filters=init_filters,
            encoder_expansion_ratio=encoder_expansion_ratio,
            decoder_expansion_ratio=decoder_expansion_ratio,
            bottleneck_expansion_ratio=bottleneck_expansion_ratio,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            use_residual_connection=True,
            norm_type=norm_type,
            global_resp_norm=False,
            blocks_down=blocks_down,
            blocks_bottleneck=blocks_bottleneck,
            blocks_up=blocks_up,
        )

    def get_model_parameters(self):
        """Retrieve stored model parameters."""
        if self.model is None:
            raise ValueError("Model has not been initialized yet. Call build_model() or bayesian_search() first.")
        
        return {
            'architecture': 'MedNeXt',
            'num_classes': self.num_classes,
            'init_filters': self.model.init_filters,
            'encoder_expansion_ratio': self.model.encoder_expansion_ratio,
            'decoder_expansion_ratio': self.model.decoder_expansion_ratio,
            'bottleneck_expansion_ratio': self.model.bottleneck_expansion_ratio,
            'kernel_size': self.model.kernel_size,
            'deep_supervision': self.model.do_ds,
            'use_residual_connection': self.model.use_residual_connection,
            'norm_type': self.model.norm_type,
            'global_resp_norm': self.model.global_resp_norm,
        }