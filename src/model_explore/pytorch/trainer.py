
from monai.transforms import AsDiscrete, Compose, Activationsd, AsDiscreted
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from typing import List, Optional
import matplotlib.pyplot as plt
import torch, os, mlflow
from tqdm import tqdm 
import numpy as np

# Not Ideal, but Necessary if Class is Missing From Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ModelTrainer:

    def __init__(self, 
                 model, 
                 device,
                 loss_function, 
                 metrics_function, 
                 optimizer):

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.metrics_function = metrics_function
        self.optimizer = optimizer

        self.parallel_mlflow = False
        self.client = None
        self.trial_run_id = None  

    def set_parallel_mlflow(self, 
                            client,
                            trial_run_id):
        
        self.parallel_mlflow = True
        self.client = client
        self.trial_run_id = trial_run_id
    
    def train_update(self):

        step = 0
        epoch_loss = 0
        self.model.train()
        for batch_data in self.train_loader:
            step += 1
            inputs = batch_data["image"].to(self.device)  # Shape: [B, C, H, W, D]
            labels = batch_data["label"].to(self.device)  # Shape: [B, C, H, W, D]            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)    # Output shape: [B, num_classes, H, W, D]          
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Update running epoch loss
            epoch_loss += loss.item()
        
        # Compute and log average epoch loss
        epoch_loss /= step
        return epoch_loss

    def validate_update(self):
        """
        Perform validation and compute metrics, including validation loss.
        """        

        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                
                # Apply sliding window inference
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=(self.crop_size, self.crop_size, self.crop_size), 
                    sw_batch_size=4,
                    predictor=self.model, 
                    overlap=0.5,
                    device=self.device
                )

                # Compute the loss for this batch
                loss = self.loss_function(val_outputs, val_labels)  # Assuming self.loss_function is defined
                val_loss += loss.item()  # Accumulate the loss                

                # Apply post-processing
                metric_val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
                metric_val_labels = [self.post_label(i) for i in decollate_batch(val_labels)]                             
                
                # Compute metrics
                self.metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)                

        # # Contains recall, precision, and f1 for each class
        metric_values = self.metrics_function.aggregate(reduction='mean_batch')

        # Compute average validation loss and add to metrics dictionary
        val_loss /= len(self.val_loader)
        metric_values.append(val_loss)

        return metric_values

    def train(
        self,
        data_load_gen,
        model_save_path: str = 'results',
        my_num_samples: int = 15,
        crop_size: int = 96,
        max_epochs: int = 100,
        val_interval: int = 15,
        lr_scheduler_type: str = 'cosine', 
        best_metric: str = 'avg_f1',
        use_mlflow: bool = False,
        verbose: bool = False
    ):

        # best lr scheduler options are cosine or reduce

        self.warmup_epochs = 5
        self.warmup_lr_factor = 0.1
        self.min_lr = 1e-7

        self.max_epochs = max_epochs
        self.crop_size = crop_size
        self.num_samples = my_num_samples
        self.val_interval = val_interval
        self.use_mlflow = use_mlflow

        # Early Stopping Parameters
        self.nan_counter = 0
        self.max_nan_epochs = 10

        # Create Save Folder if It Doesn't Exist
        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)  

        Nclass = data_load_gen.Nclasses
        self.create_results_dictionary(Nclass)  
        
        self.post_pred = AsDiscrete(argmax=True, to_onehot=Nclass)
        self.post_label = AsDiscrete(to_onehot=Nclass)                             

        # Produce Dataloaders for the First Training Iteration
        self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders(crop_size=crop_size, num_samples=my_num_samples)

        # Save the original learning rate
        original_lr = self.optimizer.param_groups[0]['lr']
        self.load_learning_rate_scheduler(lr_scheduler_type)

        # Initialize tqdm around the epoch loop
        for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):

            # Reload dataloaders periodically
            if data_load_gen.reload_frequency > 0 and (epoch + 1) % data_load_gen.reload_frequency == 0:
                self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders(num_samples=my_num_samples)
                # Lower the learning rate for the warm-up period
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = original_lr * self.warmup_lr_factor

            # Compute and log average epoch loss           
            epoch_loss = self.train_update()

            # Check for NaN in the loss
            if self.check_for_early_stopping(epoch_loss):
                tqdm.write(f"Training stopped early due to NaN values in loss for more than {self.max_nan_epochs} epochs.")
                break

            current_lr = self.optimizer.param_groups[0]['lr']
            self.my_log_metrics( metrics_dict={"loss": epoch_loss}, curr_step=epoch + 1 )
            self.my_log_metrics( metrics_dict={"learning_rate": current_lr}, curr_step=epoch + 1 )                        

            # Validation and metric logging
            if (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs:
                if verbose:
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_train_loss: {epoch_loss:.4f}")

                metric_values = self.validate_update()

                # Log all metrics
                self.my_log_metrics( metrics_dict=metric_values, curr_step=epoch + 1 )

                # Update tqdm description        
                if verbose:
                    (avg_f1, avg_recall, avg_precision) = (self.results['avg_f1'][-1][1], 
                                                           self.results['avg_recall'][-1][1], 
                                                           self.results['avg_precision'][-1][1])
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_f1_score: {avg_f1:.4f}, avg_recall: {avg_recall:.4f}, avg_precision: {avg_precision:.4f}")

                # Reset metrics function
                self.metrics_function.reset()

                # Save the best model
                if self.results[best_metric][-1][1] > self.results["best_metric"]:
                    self.results["best_metric"] = self.results[best_metric][-1][1]
                    self.results["best_metric_epoch"] = epoch + 1
                    self.model_weights = self.model.state_dict()
                    if model_save_path is not None: 
                        torch.save(self.model.state_dict(), os.path.join(model_save_path, "best_metric_model.pth"))    

                # Save plot if Local Training Call
                if not self.use_mlflow:
                    self.plot_results(save_plot=os.path.join(model_save_path, "net_train_history.png"))

            # Run the learning rate scheduler
            early_stop = self.run_scheduler(data_load_gen, original_lr, epoch, val_interval, lr_scheduler_type)
            if early_stop:
                break

        return self.results

    def load_learning_rate_scheduler(self, type: str = 'cosine'):
        """
        Initialize and return the learning rate scheduler based on the given type.
        """
        # Configure learning rate scheduler based on the type
        if type == "cosine":
            eta_min = 1e-7
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs, eta_min=eta_min )
        elif type == "onecyle":
            max_lr = 1e-3
            steps_per_epoch = len(self.train_loader)
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, epochs=self.max_epochs, steps_per_epoch=steps_per_epoch )
        elif type == "reduce":
            mode = "min"
            patience = 3
            factor = 0.5
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, patience=patience, factor=factor )
        elif type == 'exponential':
            gamma = 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler type: {type}")

    def run_scheduler(
        self, 
        data_load_gen, 
        original_lr: float,
        epoch: int,
        val_interval: int,
        type: str
        ):
        """
        Manage the learning rate scheduler, including warm-up and normal scheduling.
        """
        # Apply warm-up logic
        if (epoch + 1) <= self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = original_lr * self.warmup_lr_factor
            return False  # Continue training

        # Step the scheduler
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            self.lr_scheduler.step()
        elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and (epoch + 1) % val_interval == 0:
            metric_value = self.results['val_loss'][-1][1]
            self.lr_scheduler.step(metric_value)
        else:
            self.lr_scheduler.step()  # Step for other schedulers

        # Check learning rate for early stopping
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.min_lr and type != 'onecycle':
            print(f"Early stopping triggered at epoch {epoch + 1} as learning rate fell below {self.min_lr}.")
            return True  # Indicate early stopping

        return False  # Continue training

        # # Apply warm-up learning rate if within warm-up period
        # if (epoch + 1) % data_load_gen.reload_frequency < self.warmup_epochs:
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         # No scheduler step during warm-up for ReduceLROnPlateau
        #         return
        # else:
        #     # Restore original learning rate after warm-up
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = original_lr

        #     # Step the scheduler
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and (epoch + 1) % val_interval == 0:
        #         metric_value = self.results['val_loss'][-1][1]
        #         self.lr_scheduler.step(metric_value)  # Step with validation loss
        #     else:
        #         self.lr_scheduler.step()  # Step for other schedulers

        # # Check learning rate for early stopping
        # current_lr = self.optimizer.param_groups[0]['lr']
        # if current_lr < self.min_lr and type != 'onecycle':
        #     print(f"Early stopping triggered at epoch {epoch + 1} as learning rate fell below {self.min_lr}.")
        #     return True  # Indicate early stopping
        # return False # Continue training

    def check_for_early_stopping(self, epoch_loss: float):
        # Check for NaN in the loss
        if np.isnan(epoch_loss):
            self.nan_counter += 1
            if self.nan_counter > self.max_nan_epochs:
                return True
        else:
            self.nan_counter = 0  # Reset the counter if loss is valid
            return False

    def create_results_dictionary(self, Nclass: int):

        self.results = {
            'loss': [],
            'val_loss': [],
            'avg_f1': [],
            'avg_recall': [],
            'avg_precision': [],
            'best_metric': -1,  # Initialize as None or a default value
            'best_metric_epoch': -1
        }

        for i in range(Nclass-1):
            self.results[f'f1_class{i+1}'] = []
            self.results[f'recall_class{i+1}'] = []
            self.results[f'precision_class{i+1}'] = []

    def my_log_metrics(
        self,
        metrics_dict: dict,
        curr_step: int,
        ):

        # If metrics_dict contains multiple elements (e.g., recall, precision, f1), process them
        if len(metrics_dict) > 1:

            # Extract individual metrics 
            # (assume metrics_dict contains recall, precision, f1, val_loss in sequence)
            recall, precision, f1, val_loss = metrics_dict[0], metrics_dict[1], metrics_dict[2], metrics_dict[3]

            # Log per-class metrics
            metrics_to_log = {}
            for i, (rec, prec, f1) in enumerate(zip(recall, precision, f1)):
                metrics_to_log[f"recall_class{i+1}"] = rec.item()
                metrics_to_log[f"precision_class{i+1}"] = prec.item()
                metrics_to_log[f"f1_class{i+1}"] = f1.item()

            # Prepare average metrics
            metrics_to_log["avg_recall"] = recall.mean().cpu().item()
            metrics_to_log["avg_precision"] = precision.mean().cpu().item()
            metrics_to_log["avg_f1"] = f1.mean().cpu().item()
            metrics_to_log['val_loss'] = val_loss

            # Update metrics_dict for further logging
            metrics_dict = metrics_to_log

        # Log all metrics (per-class and average metrics)
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.results:
                self.results[metric_name] = []
            self.results[metric_name].append((curr_step, value))

        # Log to MLflow or client
        if self.client is not None and self.trial_run_id is not None:
            for metric_name, value in metrics_dict.items():
                self.client.log_metric(
                    run_id=self.trial_run_id,
                    key=metric_name,
                    value=value,
                    step=curr_step,
                )
        elif self.use_mlflow:
            for metric_name, value in metrics_dict.items():
                mlflow.log_metric(metric_name, value, step=curr_step)

    def my_log_params(
        self,
        params_dict: dict, 
        ):

        # self.results["params"] = params_dict

        # if self.client is not None and self.trial_run_id is not None:
        #     self.client.log_params(run_id=self.trial_run_id, params=params_dict)
        # elif self.use_mlflow:
        #     mlflow.log_params(params_dict)
        if self.client is not None and self.trial_run_id is not None:
            for key, value in params_dict.items():
                self.client.log_param(run_id=self.trial_run_id, key=key, value=value)
        else:
            mlflow.log_params(params_dict)                 
    
    def plot_results(self, 
                     class_names: Optional[List[str]] = None,
                     save_plot: str = None):

        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Metrics Over Epochs", fontsize=16)

        # Unpack the data for loss (logged every epoch)
        epochs_loss = [epoch for epoch, _ in self.results['loss']]
        loss = [value for _, value in self.results['loss']]
        val_epochs_loss = [epoch for epoch, _ in self.results['val_loss']]
        val_loss = [value for _,value in self.results['val_loss']]

        # Plot Training Loss in the top-left
        axs[0, 0].plot(epochs_loss, loss, label="Training Loss")
        axs[0, 0].plot(val_epochs_loss, val_loss, label='Validation Loss')
        axs[0, 0].set_xlabel("Epochs")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].legend()
        axs[0, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # For metrics that are logged every `val_interval` epochs
        epochs_metrics = [epoch for epoch, _ in self.results['avg_recall']]
        
        # Determine the number of classes and names
        num_classes = len([key for key in self.results.keys() if key.startswith('recall_class')])

        if class_names is None or len(class_names) != num_classes - 1:
            class_names = [f"Class {i+1}" for i in range(num_classes)]

        # Plot Recall in the top-right
        for class_idx in range(num_classes):
            recall_class = [value for _, value in self.results[f'recall_class{class_idx+1}']]
            axs[0, 1].plot(epochs_metrics, recall_class, label=f"{class_names[class_idx]}")
        axs[0, 1].set_xlabel("Epochs")
        axs[0, 1].set_ylabel("Recall")
        axs[0, 1].set_title("Recall per Class")
        # axs[0, 1].legend()
        axs[0, 1].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # Plot Precision in the bottom-left
        for class_idx in range(num_classes):
            precision_class = [value for _, value in self.results[f'precision_class{class_idx+1}']]
            axs[1, 0].plot(epochs_metrics, precision_class, label=f"{class_names[class_idx]}")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].set_ylabel("Precision")
        axs[1, 0].set_title("Precision per Class")
        axs[1, 0].legend()
        axs[1, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # Plot F1 Score in the bottom-right
        for class_idx in range(num_classes):
            f1_class = [value for _, value in self.results[f'f1_class{class_idx+1}']]
            axs[1, 1].plot(epochs_metrics, f1_class, label=f"{class_names[class_idx]}")
        axs[1, 1].set_xlabel("Epochs")
        axs[1, 1].set_ylabel("F1 Score")
        axs[1, 1].set_title("F1 Score per Class")
        # axs[1, 1].legend()
        axs[1, 1].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

        if save_plot: 
            fig.savefig(save_plot)
        else:
            plt.show()
