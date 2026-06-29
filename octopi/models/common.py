from monai.losses import FocalLoss, TverskyLoss, DeepSupervisionLoss
from octopi.utils import losses
from octopi.models import (
    Unet, SwinUNETR, DynUnet, SegResNet
)

def get_model(architecture):

    # Initialize model based on architecture
    if architecture == "Unet":
        model = Unet.myUNet()
    elif architecture == "SwinUNETR":
        model = SwinUNETR.mySwinUNETR()
    elif architecture == "DynUNet":
        model = DynUnet.myDynUNet()
    elif architecture == "SegResNet":
        model = SegResNet.mySegResNet()
    else:
        raise ValueError(
            f"Model type {architecture} not supported!\n"
            "Please use one of the following: Unet, SwinUNETR, DynUNet, SegResNet"
        )

    return model


def get_loss_function(trial=None, loss_name=None, gamma=None, alpha=None, weight_tversky=None):
    """Build a segmentation loss.

    Two modes:
      - Optuna search: pass ``trial`` (and optionally ``loss_name``); any param not given
        explicitly is sampled from ``trial``.
      - Explicit/config-driven: pass ``trial=None`` with ``loss_name`` and the relevant params
        (``gamma``/``alpha``/``weight_tversky``); nothing is sampled. Used by ``octopi train`` to
        rebuild the exact loss recorded in a model config's ``optimizer:`` block.
    """

    def _default_when_no_trial(name, default):
        # Config-driven mode (trial is None) but the param was omitted from the config. Fall back to
        # a documented default rather than crashing on trial.suggest_float() with no trial.
        print(f"[Warning] get_loss_function: '{name}' not provided and no Optuna trial to sample "
              f"from; using default {name}={default}.")
        return default

    def _suggest_gamma(g):
        if g is not None:
            return g
        if trial is None:
            return _default_when_no_trial("gamma", 2.0)
        return round(trial.suggest_float("gamma", 0.1, 2), 3)

    def _suggest_alpha(a):
        if a is not None:
            return a
        if trial is None:
            return _default_when_no_trial("alpha", 0.5)
        return round(trial.suggest_float("alpha", 0.1, 0.5), 3)

    def _suggest_weight_tversky(w):
        if w is not None:
            return w
        if trial is None:
            return _default_when_no_trial("weight_tversky", 0.5)
        return round(trial.suggest_float("weight_tversky", 0.1, 0.9), 3)

    # Loss function selection
    if loss_name is None:
        loss_name = trial.suggest_categorical(
            "loss_function",
            ["FocalLoss", "WeightedFocalTverskyLoss", 'FocalTverskyLoss'])

    if loss_name == "FocalLoss":
        gamma = _suggest_gamma(gamma)
        loss_function = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=gamma)

    elif loss_name == "TverskyLoss":
        alpha = _suggest_alpha(alpha)
        beta = 1.0 - alpha
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)

    elif loss_name == 'WeightedFocalTverskyLoss':
        gamma = _suggest_gamma(gamma)
        alpha = _suggest_alpha(alpha)
        beta = 1.0 - alpha
        weight_tversky = _suggest_weight_tversky(weight_tversky)
        weight_focal = 1.0 - weight_tversky
        loss_function = losses.WeightedFocalTverskyLoss(
            gamma=gamma, alpha=alpha, beta=beta,
            weight_tversky=weight_tversky, weight_focal=weight_focal
        )

    elif loss_name == 'FocalTverskyLoss':
        gamma = _suggest_gamma(gamma)
        alpha = _suggest_alpha(alpha)
        beta = 1.0 - alpha
        loss_function = losses.FocalTverskyLoss(gamma=gamma, alpha=alpha, beta=beta)

    else:
        raise ValueError(f"Unsupported loss_function '{loss_name}'.")

    return loss_function

def get_default_unet_params() -> dict:
    """
    Returns the default parameters for a UNet model.
    """
    model_config = {
        'architecture': 'Unet',
        'strides': [2, 2, 1],
        'channels': [48, 64, 80, 80],
        'dropout': 0.0, 'num_res_units': 1,
    }
    return model_config


####################### Deep Supervision Utility Functions #######################

def uses_deep_supervision(model_config: dict) -> bool:
    """Check whether a model config enables deep supervision."""
    arch = model_config.get('architecture', '')
    if arch == 'DynUNet':
        return model_config.get('deep_supervision', False)
    if arch == 'SegResNet':
        return model_config.get('dsdepth', 1) > 1
    return False


def wrap_loss_for_ds(loss_function, model_config: dict, weight_mode: str = "exp"):
    """Wrap a loss function with DeepSupervisionLoss if the model uses DS.

    Args:
        loss_function: Base loss function.
        model_config: Model configuration dict.
        weight_mode: Weighting scheme for DS levels ('exp', 'same', 'two').

    Returns:
        The original loss_function (unchanged) if DS is off, or a
        DeepSupervisionLoss wrapper if DS is on.
    """
    if not uses_deep_supervision(model_config):
        return loss_function
    return DeepSupervisionLoss(loss=loss_function, weight_mode=weight_mode)