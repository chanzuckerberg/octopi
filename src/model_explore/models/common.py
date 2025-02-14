from monai.losses import FocalLoss, TverskyLoss
from model_explore import losses
from model_explore.models import (
    Unet, AttentionUnet
)

def get_loss_function(trial, loss_name = None):

    # Loss function selection
    if loss_name is None:
        loss_name = trial.suggest_categorical(
            "loss_function", 
            ["FocalLoss", "TverskyLoss", 'WeightedFocalTverskyLoss', 'FocalTverskyLoss'])

    if loss_name == "FocalLoss":
        gamma = round(trial.suggest_float("gamma", 0.1, 4), 3)
        loss_function = FocalLoss(include_background=True, to_onehot_y=True, use_softmax=True, gamma=gamma)

    elif loss_name == "TverskyLoss":
        alpha = round(trial.suggest_float("alpha", 0.15, 0.75), 3)
        beta = 1.0 - alpha
        loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)

    elif loss_name == 'WeightedFocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 4), 3)
        alpha = round(trial.suggest_float("alpha", 0.15, 0.75), 3)
        beta = 1.0 - alpha
        weight_tversky = round(trial.suggest_float("weight_tversky", 0.1, 0.9), 3)
        weight_focal = 1.0 - weight_tversky
        loss_function = losses.WeightedFocalTverskyLoss(
            gamma=gamma, alpha=alpha, beta=beta,
            weight_tversky=weight_tversky, weight_focal=weight_focal
        )

    elif loss_name == 'FocalTverskyLoss':
        gamma = round(trial.suggest_float("gamma", 0.1, 4.0), 3)
        alpha = round(trial.suggest_float("alpha", 0.15, 0.85), 3)
        beta = 1.0 - alpha
        loss_function = losses.FocalTverskyLoss(gamma=gamma, alpha=alpha, beta=beta)

    return loss_function

def get_model(num_classes, device, model_type):

    if model_type == "Unet":
        model = Unet.myUNet(num_classes, device)
    elif model_type == "AttentionUnet":
        model = AttentionUnet.myAttentionUnet(num_classes, device)

    return model


#### TODO : Models to try Adding? 
# 1. SWIN UNETR 
# 2. SegResNet / SegResNetDS 
# 3. MedNext 
# 4. Swin-Conv-UNet