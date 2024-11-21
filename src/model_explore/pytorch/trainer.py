import model_explore.pytorch.my_metrics as metrics
from typing import Dict, List, Optional, Tuple
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch, os

class unet:

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

        self.model.train()
        epoch_loss = 0
        step = 0
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

        self.model.eval()
        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                val_outputs = self.model(val_inputs)
                
                # Decollate batches into lists
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                
                # Apply post-processing
                metric_val_outputs = [self.post_pred(i) for i in val_outputs_list]
                metric_val_labels = [self.post_label(i) for i in val_labels_list]
                
                # Compute metrics
                self.metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)

        # Contains recall, precision, and f1 for each class
        if self.parallel_mlflow == False:
            metric_values = self.metrics_function.aggregate(reduction='mean_batch')
        else:
            # Metrics Function is a list with length of [nGPU x nClass]
            f_local, _ = metrics.do_metric_reduction(self.metrics_function._buffers[0], 
                                                     reduction="mean_batch", 
                                                     target_device = self.device )
            metric_values = []                    
            for metric_name in self.metrics_function.metric_name:
                metric = metrics.compute_confusion_matrix_metric(metric_name, f_local)
                metric_values.append(metric.cpu())

        return metric_values


    def mlflow_train(self,
                     data_load_gen, 
                     my_num_samples: int = 15,                     
                     max_epochs: int = 100,
                     crop_size: int = 96,                     
                     val_interval: int = 15,                 
                     model_save_path: str = None,
                     verbose: bool = False):
        
        self.crop_size = crop_size
        self.num_samples = my_num_samples
        self.val_interval = val_interval

        # Create Save Folder if It Doesn't Exist
        if model_save_path is not None and not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True) 

        Nclass = data_load_gen.Nclasses
        results = self.create_results_dictionary(Nclass)         
        self.post_pred = AsDiscrete(argmax=True, to_onehot=Nclass)
        self.post_label = AsDiscrete(to_onehot=Nclass)  

        best_f1_metric = -1
        best_metric_epoch = -1

        # Produce Dataloaders for the First Training Iteration
        self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders( crop_size = crop_size, num_samples = my_num_samples )                

        # Initialize tqdm around the epoch loop
        for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):

            # Load new tomograms every few epochs 
            if data_load_gen.reload_frequency > 0 and (epoch + 1) % data_load_gen.reload_frequency == 0:
                self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders( num_samples = my_num_samples )

            # Compute and log average epoch loss
            epoch_loss = self.train_update()
            results['loss'].append((epoch + 1, epoch_loss))            
            metrics.my_log_metric(f"train_loss", epoch_loss, epoch + 1, self.client, self.trial_run_id)

            if (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs:

                # Update tqdm description with dynamic loss for each epoch
                if verbose: tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_train_loss: {epoch_loss:.4f}")

                metric_values = self.validate_update()

                # Log per-class metrics
                for i, (rec, prec, f1) in enumerate(zip(metric_values[0], metric_values[1], metric_values[2])):
                    metrics.my_log_metric(f"recall_class_{i+1}", rec.item(), epoch + 1, self.client, self.trial_run_id)
                    metrics.my_log_metric(f"precision_class_{i+1}", prec.item(), epoch + 1, self.client, self.trial_run_id)
                    metrics.my_log_metric(f"f1_score_class_{i+1}", f1.item(), epoch + 1, self.client, self.trial_run_id)                    

                # Log average metrics across all classes
                avg_recall =  metric_values[0].mean()
                avg_precision =  metric_values[1].mean()
                avg_f1 =  metric_values[2].mean()                                

                metrics.my_log_metric(f"avg_recall", avg_recall, epoch + 1, self.client, self.trial_run_id)
                metrics.my_log_metric(f"avg_precision", avg_precision, epoch + 1, self.client, self.trial_run_id)
                metrics.my_log_metric(f"avg_f1_score", avg_f1, epoch + 1, self.client, self.trial_run_id)      

                if verbose: 
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_recall: {avg_recall:.4f}")
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_precision: {avg_precision:.4f}")
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_f1_score: {avg_f1:.4f}")

                # tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_recall: {avg_recall:.4f}")        
                self.metrics_function.reset()

                # # Track the best F1 score (mean across all classes)
                if avg_f1 > best_f1_metric:
                    best_f1_metric = avg_f1
                    best_metric_epoch = epoch + 1
                    if model_save_path is not None:
                        torch.save(self.model.state_dict(), os.path.join(model_save_path, "best_metric_model.pth"))

        return best_f1_metric, best_metric_epoch

    # Instead of Logging with ML-Flow, I want to return a dictionary with all the metrics
    def local_train(self,
                     data_load_gen, 
                     model_save_path = 'results',
                     my_num_samples: int = 15,
                     crop_size: int = 96,
                     max_epochs: int = 100,
                     val_interval: int = 15,
                     verbose: bool = False):
        
        self.val_interval = val_interval
        self.num_samples = my_num_samples   
        self.crop_size = crop_size

        # Create Save Folder if It Doesn't Exist
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path) 

        Nclass = data_load_gen.Nclasses
        results = self.create_results_dictionary(Nclass)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=Nclass)
        self.post_label = AsDiscrete(to_onehot=Nclass)  

        # Produce Dataloaders for the First Training Iteration
        self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders( crop_size = crop_size, num_samples = my_num_samples)        

        # Initialize tqdm around the epoch loop
        for epoch in tqdm(range(max_epochs), desc="Training Progress", unit="epoch"):

            # Load new tomograms every few epochs 
            if data_load_gen.reload_frequency > 0 and (epoch + 1) % data_load_gen.reload_frequency == 0:
                self.train_loader, self.val_loader = data_load_gen.create_train_dataloaders( num_samples = my_num_samples )

            # Compute and log average epoch loss
            epoch_loss = self.train_update()
            results['loss'].append((epoch + 1, epoch_loss))

            if (epoch + 1) % val_interval == 0:

                # Update tqdm description with dynamic loss for each epoch
                if verbose: tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_train_loss: {epoch_loss:.4f}")

                metric_values = self.validate_update()

                # Log per-class metrics
                for i, (rec, prec, f1) in enumerate(zip(metric_values[0], metric_values[1], metric_values[2])):
                    
                    # Append each class metric to the results dictionary with epoch
                    results[f'recall_class{i+1}'].append((epoch + 1, rec.item()))
                    results[f'precision_class{i+1}'].append((epoch + 1, prec.item()))
                    results[f'f1_class{i+1}'].append((epoch + 1, f1.item()))                    

                # Log average metrics across all classes
                avg_recall =  metric_values[0].mean().cpu().item()
                avg_precision =  metric_values[1].mean().cpu().item()
                avg_f1 =  metric_values[2].mean().cpu().item()

                results['avg_recall'].append((epoch + 1, avg_recall))
                results['avg_precision'].append((epoch + 1, avg_precision))
                results['avg_f1'].append((epoch + 1, avg_f1))

                if verbose: 
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_recall: {avg_recall:.4f}")
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_precision: {avg_precision:.4f}")
                    tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_f1_score: {avg_f1:.4f}")

                # tqdm.write(f"Epoch {epoch + 1}/{max_epochs}, avg_recall: {avg_recall:.4f}")        
                self.metrics_function.reset()

                # # Track the best F1 score (mean across all classes)
                if avg_f1 > results['best_metric']:
                    results['best_metric'] = avg_f1
                    results['best_metric_epoch'] = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(model_save_path, "best_metric_model.pth"))

                # Save Plots to model_save_path: 
                plot_save_path = os.path.join(model_save_path, 'net_train_history.png')
                self.plot_results(results, save_plot = plot_save_path)

        return results
    
    def create_results_dictionary(self, Nclass: int):

        results = {
            'loss': [],
            'avg_f1': [],
            'avg_recall': [],
            'avg_precision': [],
            'best_metric': -1,  # Initialize as None or a default value
            'best_metric_epoch': -1
        }

        for i in range(Nclass):
            results[f'f1_class{i+1}'] = []
            results[f'recall_class{i+1}'] = []
            results[f'precision_class{i+1}'] = []

        return results
    
    def plot_results(self, 
                     results: Dict[str, List[Tuple[int, float]]],
                     class_names: Optional[List[str]] = None,
                     save_plot: str = None):

        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Metrics Over Epochs", fontsize=16)

        # Unpack the data for loss (logged every epoch)
        epochs_loss = [epoch for epoch, _ in results['loss']]
        loss = [value for _, value in results['loss']]

        # Plot Training Loss in the top-left
        axs[0, 0].plot(epochs_loss, loss, label="Training Loss")
        axs[0, 0].set_xlabel("Epochs")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].legend()
        axs[0, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # For metrics that are logged every `val_interval` epochs
        epochs_metrics = [epoch for epoch, _ in results['avg_recall']]
        
        # Determine the number of classes and names
        num_classes = len([key for key in results.keys() if key.startswith('recall_class')])

        if class_names is None or len(class_names) != num_classes - 1:
            class_names = [f"Class {i+1}" for i in range(num_classes)]

        # Plot Recall in the top-right
        for class_idx in range(1, num_classes):
            recall_class = [value for _, value in results[f'recall_class{class_idx}']]
            axs[0, 1].plot(epochs_metrics, recall_class, label=f"{class_names[class_idx-1]}")
        axs[0, 1].set_xlabel("Epochs")
        axs[0, 1].set_ylabel("Recall")
        axs[0, 1].set_title("Recall per Class")
        # axs[0, 1].legend()
        axs[0, 1].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # Plot Precision in the bottom-left
        for class_idx in range(1, num_classes):
            precision_class = [value for _, value in results[f'precision_class{class_idx}']]
            axs[1, 0].plot(epochs_metrics, precision_class, label=f"{class_names[class_idx-1]}")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].set_ylabel("Precision")
        axs[1, 0].set_title("Precision per Class")
        axs[1, 0].legend()
        axs[1, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

        # Plot F1 Score in the bottom-right
        for class_idx in range(1, num_classes):
            f1_class = [value for _, value in results[f'f1_class{class_idx}']]
            axs[1, 1].plot(epochs_metrics, f1_class, label=f"{class_names[class_idx-1]}")
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
