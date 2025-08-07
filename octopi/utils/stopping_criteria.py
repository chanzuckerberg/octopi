import numpy as np

class EarlyStoppingChecker:
    """
    A class to manage various early stopping criteria for model training.
    """
    
    def __init__(self, 
                 max_nan_epochs=15, 
                 plateau_patience=10, 
                 plateau_min_delta=0.005,
                 stagnation_patience=30, 
                 convergence_window=5, 
                 convergence_threshold=0.005,
                 val_interval=15,
                 monitor_metric='avg_fbeta'):
        """
        Initialize early stopping parameters.
        
        Args:
            max_nan_epochs: Maximum number of epochs with NaN loss before stopping
            plateau_patience: Number of validation checks to wait for plateau detection
            plateau_min_delta: Minimum change to qualify as improvement
            stagnation_patience: Number of validation intervals to wait for best metric improvement
            convergence_window: Window size for calculating improvement rate
            convergence_threshold: Minimum improvement rate threshold
            val_interval: Number of epochs between validation runs
            monitor_metric: Primary metric to monitor for early stopping criteria
        """
        self.max_nan_epochs = max_nan_epochs
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.stagnation_patience = stagnation_patience
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.val_interval = val_interval
        self.monitor_metric = monitor_metric
        
        # Counters
        self.nan_counter = 0
        
        # Flags for detailed reporting
        self.stopped_reason = None

    def check_for_nan(self, epoch_loss):
        """Check for NaN in the loss."""
        if np.isnan(epoch_loss):
            self.nan_counter += 1
            if self.nan_counter > self.max_nan_epochs:
                self.stopped_reason = f"NaN values in loss for more than {self.max_nan_epochs} epochs"
                return True
        else:
            self.nan_counter = 0  # Reset the counter if loss is valid
        return False

    def check_for_plateau(self, results):
        """Detect plateaus in validation metrics."""
        if len(results[self.monitor_metric]) < self.plateau_patience + 1:
            return False
            
        # Get the last 'patience' number of validation points
        recent_values = [x[1] for x in results[self.monitor_metric][-self.plateau_patience:]]
        # Find the max value in the window
        max_value = max(recent_values)
        # Find the min value in the window
        min_value = min(recent_values)
        
        # If the range of values is small, consider it a plateau
        if max_value - min_value < self.plateau_min_delta:
            self.stopped_reason = f"{self.monitor_metric} plateaued for {self.plateau_patience} validations"
            return True
        
        return False

    def check_best_metric_stagnation(self, results):
        """Stop if best metric hasn't improved for a number of validation intervals."""
        if "best_metric_epoch" not in results or len(results[self.monitor_metric]) < self.stagnation_patience + 1:
            return False
            
        # Get epoch of the best metric so far
        best_epoch = results["best_metric_epoch"]
        current_epoch = results[self.monitor_metric][-1][0]
        
        # Check if it's been more than 'patience' validation intervals
        if (current_epoch - best_epoch) >= (self.stagnation_patience * self.val_interval):
            self.stopped_reason = f"No improvement for {self.stagnation_patience} validation intervals"
            return True
            
        return False

    # def check_convergence_rate(self, results):
    #     """Stop when improvement rate slows below threshold."""
    #     if len(results[self.monitor_metric]) < self.convergence_window + 1:
    #         return False
        
    #     # Calculate average improvement rate over window
    #     recent_values = [x[1] for x in results[self.monitor_metric][-(self.convergence_window+1):]]
    #     improvements = [recent_values[i+1] - recent_values[i] for i in range(self.convergence_window)]
    #     avg_improvement = sum(improvements) / self.convergence_window
        
    #     if avg_improvement < self.convergence_threshold and avg_improvement > 0:
    #         self.stopped_reason = f"Convergence rate ({avg_improvement:.6f}) below threshold"
    #         return True
            
    #     return False

    def should_stop_training(self, epoch_loss, results=None, check_metrics=False):
        """
        Comprehensive check for whether training should stop.
        
        Args:
            epoch_loss: Current epoch's loss value
            results: Dictionary containing training metrics history
            check_metrics: Whether to also check validation metrics-based criteria
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check for NaN in loss (can be done every epoch)
        if self.check_for_nan(epoch_loss):
            return True
            
        # Only check metric-based criteria if requested and results are provided
        if check_metrics and results:
            # Check for plateau in validation metrics
            if self.check_for_plateau(results):
                return True
                
            # Check if best metric hasn't improved for a while
            if self.check_best_metric_stagnation(results):
                return True
                
            # # Check if convergence rate has slowed down
            # if self.check_convergence_rate(results):
            #     return True
                
        return False
    
    def get_stopped_reason(self):
        """Get the reason for stopping, if any."""
        if self.stopped_reason:
            return f"Early stopping triggered: {self.stopped_reason}"
        return "No early stopping criteria met."