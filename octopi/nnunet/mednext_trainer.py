"""
nnUNet v2 compatible trainers for MedNeXt (mednextv1 package).

Subclasses nnUNetTrainer and overrides build_network_architecture() to
instantiate MedNeXt in place of nnUNet's default PlainConvUNet.

This file is auto-copied into the nnunetv2 trainer directory by
octopi nnunet train when a mednext model is selected.

Requires: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git

Variant specs (from create_mednext_v1.py):
  S — block_counts [2,2,2,2,2,2,2,2,2], exp_r 2 (scalar)
  B — block_counts [2,2,2,2,2,2,2,2,2], exp_r [2,3,4,4,4,4,4,3,2]
  M — block_counts [3,4,4,4,4,4,4,4,3], exp_r [2,3,4,4,4,4,4,3,2], grad checkpointing
  L — block_counts [3,4,8,8,8,8,8,4,3], exp_r [3,4,8,8,8,8,8,4,3], grad checkpointing
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _build_mednext(num_input_channels, num_output_channels, enable_deep_supervision,
                   kernel_size, block_counts, exp_r, checkpoint_style=None):
    from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_output_channels,
        exp_r=exp_r,
        kernel_size=kernel_size,
        deep_supervision=enable_deep_supervision,
        do_res=True,
        do_res_up_down=True,
        block_counts=block_counts,
        checkpoint_style=checkpoint_style,
    )


class _MedNeXtTrainerBase(nnUNetTrainer):
    """Mixin that fixes deep_supervision toggling for MedNeXt models.

    nnUNetTrainer.set_deep_supervision_enabled() assumes the network has a
    .decoder sub-module, but MedNeXt exposes deep_supervision directly on the
    top-level model object.
    """

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.deep_supervision = enabled


# ── Small (S) ────────────────────────────────────────────────────────────────

class nnUNetTrainerMedNeXtS_kernel3(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=3,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            exp_r=2,
        )


class nnUNetTrainerMedNeXtS_kernel5(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=5,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            exp_r=2,
        )


# ── Base (B) ─────────────────────────────────────────────────────────────────

class nnUNetTrainerMedNeXtB_kernel3(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=3,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        )


class nnUNetTrainerMedNeXtB_kernel5(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=5,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        )


# ── Medium (M) ───────────────────────────────────────────────────────────────

class nnUNetTrainerMedNeXtM_kernel3(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=3,
            block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
            checkpoint_style='outside_block',
        )


class nnUNetTrainerMedNeXtM_kernel5(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=5,
            block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
            checkpoint_style='outside_block',
        )


# ── Large (L) ────────────────────────────────────────────────────────────────

class nnUNetTrainerMedNeXtL_kernel3(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=3,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
        )


class nnUNetTrainerMedNeXtL_kernel5(_MedNeXtTrainerBase):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return _build_mednext(
            num_input_channels, num_output_channels, enable_deep_supervision,
            kernel_size=5,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
        )
