from monai.utils import (MetricReduction, look_up_option)
from monai.metrics import confusion_matrix as monai_cm
from typing import Any
import torch, mlflow

##############################################################################################################################

def do_metric_reduction(
    f_list: list[torch.Tensor], reduction: MetricReduction | str = MetricReduction.MEAN, target_device='cuda:0'
    ) -> tuple[torch.Tensor | Any, torch.Tensor]:
    """
    Perform metric reduction on a list of tensors locally on a single GPU.

    Args:
        f_list: list of tensors containing metric scores per batch and per class.
        reduction: type of reduction to apply to the metrics.
        target_device: optional device to which all tensors will be moved for reduction, defaults to current GPU.

    Returns:
        Reduced tensor with per-class metrics on the local GPU.
    """

    # Move all tensors to the target device and concatenate
    f = torch.cat([x.to(target_device) for x in f_list], dim=0)  # Concatenate along batch dimension

    # Reshape to separate batch and class dimensions, assuming each has shape [n_classes, 4]
    n_classes = f_list[0].shape[0]  # Number of classes inferred from shape [7, 4]
    f = f.view(-1, n_classes, 4)  # New shape is [batch_size, n_classes, 4]

    # Now perform the reduction as usual on a single device
    nans = torch.isnan(f)
    not_nans = ~nans

    t_zero = torch.zeros(1, device=f.device, dtype=torch.float)
    reduction = look_up_option(reduction, MetricReduction)

    if reduction == MetricReduction.NONE:
        return f, not_nans.float()
    
    import pdb; pdb.set_trace()

    f[nans] = 0  # Set NaNs to zero for reduction
    if reduction == MetricReduction.MEAN:
        not_nans = not_nans.sum(dim=1).float()
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)
        not_nans = (not_nans > 0).sum(dim=0).float()
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)
    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=[0, 1]).float()
        f = f.sum(dim=[0, 1])
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0).float()
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0).float()
        f = f.sum(dim=0)
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1).float()
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1).float()
        f = f.sum(dim=1)
    else:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans

######################################################################################################################

def compute_confusion_matrix_metric(metric_name: str, 
                                    confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        confusion_matrix: Please see the doc string of the function ``get_confusion_matrix`` for more details.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

    device = confusion_matrix.device
    metric = monai_cm.check_confusion_matrix_metric_name(metric_name)

    input_dim = confusion_matrix.ndimension()
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")

    tp = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    tn = confusion_matrix[..., 2]
    fn = confusion_matrix[..., 3]
    p = tp + fn
    n = fp + tn
    # calculate metric
    numerator: torch.Tensor
    denominator: torch.Tensor | float
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator

##############################################################################################################################

def my_log_param(params_dict, client = None, trial_run_id = None):

    if client is not None and trial_run_id is not None:
        client.log_params(run_id=trial_run_id, params=params_dict)
    else:
        mlflow.log_params(params_dict)


##############################################################################################################################

def my_log_metric(metric_name, val, curr_step, client = None, trial_run_id = None):

    if client is not None and trial_run_id is not None:
        client.log_metric(run_id = trial_run_id, 
                          key = metric_name,
                          value = val, 
                          step = curr_step)
    else:
        mlflow.log_metric(metric_name, val, step = curr_step)