from monai.losses import FocalLoss, TverskyLoss
import torch

class WeightedFocalTverskyLoss(torch.nn.Module):
    def __init__(
        self, gamma=1.0, alpha=0.7, beta=0.3, 
        weight_tversky=0.5, weight_focal=0.5, 
        smooth=1e-5, **kwargs ):
        """
        Weighted combination of Focal and Tversky loss.

        Args:
            gamma (float): Focus parameter for Focal Loss.
            alpha (float): Weight for false positives in Tversky Loss.
            beta (float): Weight for false negatives in Tversky Loss.
            weight_tversky (float): Weight of Tversky loss in the combination.
            weight_focal (float): Weight of Focal loss in the combination.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.tversky_loss = TverskyLoss(
            alpha=alpha, beta=beta, include_background=True, 
            to_onehot_y=True, softmax=True,
            smooth_nr=smooth, smooth_dr=smooth, **kwargs
        )
        self.focal_loss = FocalLoss(
            include_background=True, to_onehot_y=True, 
            use_softmax=True, gamma=gamma
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal

    def forward(self, y_pred, y_true):
        """
        Compute the combined loss.

        Args:
            y_pred (Tensor): Predicted probabilities (B, C, ...).
            y_true (Tensor): Ground truth labels (B, C, ...), one-hot encoded.

        Returns:
            Tensor: Weighted combination of Tversky and Focal losses.
        """
        tversky = self.tversky_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        return self.weight_tversky * tversky + self.weight_focal * focal
    
class FocalTverskyLoss(TverskyLoss):
    def __init__(
        self, 
        alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-5, **kwargs):
        """
        Focal Tversky Loss with an additional power term for harder samples.

        From https://arxiv.org/abs/1810.07842

        Args:
            alpha (float): Weight for false positives.
            beta (float): Weight for false negatives.
            gamma (float): Focus parameter (like Focal Loss).
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super().__init__(
            alpha=alpha, beta=beta, 
            include_background=True, 
            to_onehot_y=True, softmax=True, 
            smooth_nr=smooth, smooth_dr=smooth, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): Predicted probabilities (B, C, ...).
            y_true (Tensor): Ground truth labels (B, C, ...), one-hot encoded.

        Returns:
            Tensor: Loss value.
        """
        tversky_loss = super().forward(y_pred, y_true)
        modified_loss = torch.pow(tversky_loss, 1 / self.gamma)
        return modified_loss
