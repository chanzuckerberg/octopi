from monai.losses import FocalLoss, TverskyLoss
from octopi.utils import losses
from octopi.models import (
    Unet, AttentionUnet, MedNeXt, SegResNet
)

def get_model(architecture):

    # Initialize model based on architecture
    if architecture == "Unet":
        model = Unet.myUNet()
    elif architecture == "AttentionUnet":
        model = AttentionUnet.myAttentionUnet()
    elif architecture == "MedNeXt":
        model = MedNeXt.myMedNeXt()
    elif architecture == "SegResNet":
        model = SegResNet.mySegResNet()
    else:
        raise ValueError(f"Model type {architecture} not supported!\nPlease use one of the following: Unet, AttentionUnet, MedNeXt, SegResNet")

    return model

def get_loss_function(trial, loss_name = None):

    # Loss function selection
    if loss_name is None:
        loss_name = trial.suggest_categorical(
            "loss_function", 
            ["FocalLoss", "WeightedFocalTverskyLoss", 'FocalTverskyLoss'])

    if loss_name == "FocalLoss":
        gamma = round(trial.suggest_float("gamma", 0.1, 2), 3)
        loss_function = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=gamma)

    elif loss_name == "TverskyLoss":
        alpha = round(trial.suggest_float("alpha", 0.1, 0.5), 3)
        beta = 1.0 - alpha
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)

    elif loss_name == 'WeightedFocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 2), 3)
        alpha = round(trial.suggest_float("alpha", 0.1, 0.5), 3)
        beta = 1.0 - alpha
        weight_tversky = round(trial.suggest_float("weight_tversky", 0.1, 0.9), 3)
        weight_focal = 1.0 - weight_tversky
        loss_function = losses.WeightedFocalTverskyLoss(
            gamma=gamma, alpha=alpha, beta=beta,
            weight_tversky=weight_tversky, weight_focal=weight_focal
        )

    elif loss_name == 'FocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 2), 3)
        alpha = round(trial.suggest_float("alpha", 0.1, 0.5), 3)
        beta = 1.0 - alpha
        loss_function = losses.FocalTverskyLoss(gamma=gamma, alpha=alpha, beta=beta)

    return loss_function


#### TODO : Models to try Adding? 
# 1. Swin UNETR 
# 2. Swin-Conv-UNet