from octopi.utils import visualization_tools as viz
from monai.inferers import sliding_window_inference
from octopi.utils import stopping_criteria
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import torch, os, mlflow, re
import torch_ema as ema
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
                 optimizer, 
                 use_ema: bool = True):

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.metrics_function = metrics_function
        self.optimizer = optimizer

        self.parallel_mlflow = False
        self.client = None
        self.trial_run_id = None  

        # Default F-Beta Value
        self.beta = 2

        # Initialize EMAHandler for the model
        self.ema_experiment = use_ema
        if self.ema_experiment:
            self.ema_handler = ema.ExponentialMovingAverage(self.model.parameters(), decay=0.99)

        # Initialize Figure and Axes for Plotting
        self.fig = None; self.axs = None

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

            # Update EMA weights
            if self.ema_experiment:
                self.ema_handler.update()

            # Update running epoch loss
            epoch_loss += loss.item()
        
        # Compute and log average epoch loss
        epoch_loss /= step
        return epoch_loss

    def validate_update(self):
        """
        Perform validation and compute metrics, including validation loss.
        """        

        # Set model to evaluation mode
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                
                # Apply sliding window inference
                # roi_size=self.input_dim, # try setting a set size of 128, 144 or 160? 
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=(144,144,144),
                    sw_batch_size=4,
                    predictor=self.model, 
                    overlap=0.5,
                    device=self.device
                )

                del val_inputs
                torch.cuda.empty_cache()

                # Compute the loss for this batch
                loss = self.loss_function(val_outputs, val_labels)  # Assuming self.loss_function is defined
                val_loss += loss.item()  # Accumulate the loss                

                # Apply post-processing
                metric_val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
                metric_val_labels = [self.post_label(i) for i in decollate_batch(val_labels)]                             
                
                # Compute metrics
                self.metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)             

                del val_labels, val_outputs, metric_val_outputs, metric_val_labels
                torch.cuda.empty_cache()

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
        self.min_lr = 1e-6

        self.max_epochs = max_epochs
        self.crop_size = crop_size
        self.num_samples = my_num_samples
        self.val_interval = val_interval
        self.use_mlflow = use_mlflow

        # Create Save Folder if It Doesn't Exist
        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)  

        Nclass = data_load_gen.Nclasses
        self.create_results_dictionary(Nclass)  

        # Resolve the best metric
        best_metric = self.resolve_best_metric(best_metric)

        # Stopping Criteria
        self.stopping_criteria = stopping_criteria.EarlyStoppingChecker(monitor_metric=best_metric, val_interval=val_interval)            

        self.post_pred = AsDiscrete(argmax=True, to_onehot=Nclass)
        self.post_label = AsDiscrete(to_onehot=Nclass)                             

        # Produce Dataloaders for the First Training Iteration
        self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders(crop_size=crop_size, num_samples=my_num_samples)
        self.input_dim = data_load_gen.input_dim

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
            if self.stopping_criteria.should_stop_training(epoch_loss):
                tqdm.write(f"Training stopped early due to {self.stopping_criteria.get_stopped_reason()}")
                break

            current_lr = self.optimizer.param_groups[0]['lr']
            self.my_log_metrics( metrics_dict={"loss": epoch_loss}, curr_step=epoch + 1 )
            self.my_log_metrics( metrics_dict={"learning_rate": current_lr}, curr_step=epoch + 1 )                        

            # Validation and metric logging
            if (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs:
                if verbose:
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_train_loss: {epoch_loss:.4f}")

                # Validate the Model with or without EMA
                if self.ema_experiment:
                    with self.ema_handler.average_parameters():
                        metric_values = self.validate_update()
                else:
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

                    # Read Model Weights and Save
                    if self.ema_experiment:
                        with self.ema_handler.average_parameters():
                            self.save_model(model_save_path)
                    else:
                        self.save_model(model_save_path)

                # Save plot if Local Training Call
                if not self.use_mlflow:
                    self.fig, self.axs = viz.plot_training_results(
                        self.results, 
                        save_plot=os.path.join(model_save_path, "net_train_history.png"), 
                        fig=self.fig, 
                        axs=self.axs)

                # After Validation Metrics are Logged, Check for Early Stopping
                if self.stopping_criteria.should_stop_training(epoch_loss, results=self.results, check_metrics=True):
                    tqdm.write(f"Training stopped early due to {self.stopping_criteria.get_stopped_reason()}")
                    break

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
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs, eta_min=self.min_lr )
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
        
    def save_model(self, model_save_path: str):

        # Store Model Weights as Member Variable
        self.model_weights = self.model.state_dict()

        # Save Model Weights to *.pth file
        if model_save_path is not None:
            torch.save(self.model_weights, os.path.join(model_save_path, "best_model.pth"))

    def create_results_dictionary(self, Nclass: int):

        self.results = {
            'loss': [],
            'val_loss': [],
            'avg_f1': [],
            'avg_recall': [],
            'avg_precision': [],
            'avg_fbeta': [],
            'best_metric': -1,  # Initialize as None or a default value
            'best_metric_epoch': -1
        }

        for i in range(Nclass-1):
            self.results[f'fbeta_class{i+1}'] = []
            self.results[f'f1_class{i+1}'] = []
            self.results[f'recall_class{i+1}'] = []
            self.results[f'precision_class{i+1}'] = []

        self.metric_names = self.results.keys()

    def my_log_metrics(
        self,
        metrics_dict: dict,
        curr_step: int,
        ):

        # If metrics_dict contains multiple elements (e.g., recall, precision, f1), process them
        if len(metrics_dict) > 1:

            # Extract individual metrics 
            # (assume metrics_dict contains recall, precision, f1, val_loss in sequence)
            recall, precision, f1s, val_loss = metrics_dict[0], metrics_dict[1], metrics_dict[2], metrics_dict[3]

            # Log per-class metrics
            metrics_to_log = {}
            for i, (rec, prec, f1) in enumerate(zip(recall, precision, f1s)):
                metrics_to_log[f"recall_class{i+1}"] = rec.item()
                metrics_to_log[f"precision_class{i+1}"] = prec.item()
                metrics_to_log[f"f1_class{i+1}"] = f1.item()
                metrics_to_log[f"fbeta_class{i+1}"] = self.fbeta(prec, rec).item()

            # Prepare average metrics
            metrics_to_log["avg_recall"] = recall.mean().cpu().item()
            metrics_to_log["avg_precision"] = precision.mean().cpu().item()
            metrics_to_log["avg_f1"] = f1s.mean().cpu().item()
            metrics_to_log["avg_fbeta"] = self.fbeta(precision, recall).mean().cpu().item()
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

    def fbeta(self, precision, recall):

        # Handle division by zero
        numerator = (1 + self.beta**2) * (precision * recall)
        denominator = (self.beta**2 * precision) + recall
        
        # Use torch.where to handle zero cases
        result = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(precision)
        )
        return result

    def my_log_params(
        self,
        params_dict: dict, 
        ):

        if self.client is not None and self.trial_run_id is not None:
            for key, value in params_dict.items():
                self.client.log_param(run_id=self.trial_run_id, key=key, value=value)
        else:
            mlflow.log_params(params_dict)  

    # Example input: best_metric = 'fBeta2_class3' or 'fBeta1' or 'f1_class2'
    def resolve_best_metric(self, best_metric):
        fbeta_pattern = r"^fBeta(\d+)(?:_class(\d+))?$"  # Matches fBetaX or fBetaX_classY
        match = re.match(fbeta_pattern, best_metric)

        if match:
            self.beta = int(match.group(1))  # Extract beta value
            class_part = match.group(2)
            if class_part:
                best_metric = f'fbeta_class{class_part}'  # fBeta2_class3 → fbeta_class3
            else:
                best_metric = 'avg_fbeta'  # fBeta2 → avg_fbeta

        elif best_metric in self.metric_names:
            pass  # It's already a valid metric in the results dict

        else:
            print(f"'{best_metric}' is not a valid metric. Defaulting to 'avg_f1'.\n")
            best_metric = 'avg_f1'

        return best_metric
    
