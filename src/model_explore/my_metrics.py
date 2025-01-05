from monai.utils import (MetricReduction, look_up_option)
from monai.metrics import confusion_matrix as monai_cm
from typing import Any
import torch, mlflow

def my_log_param(params_dict, client = None, trial_run_id = None):

    if client is not None and trial_run_id is not None:
        # client.log_params(run_id=trial_run_id, params=params_dict)
        for key, value in params_dict.items():
            client.log_param(run_id=trial_run_id, key=key, value=value)
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