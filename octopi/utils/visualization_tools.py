from ipywidgets import interact, IntSlider, fixed
from copick_utils.io import readers
import matplotlib.colors as mcolors
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import copick

# Define the interactive function
def interact_3d_seg(vol, seg):
    """
    Interactively show the segmentation on a tomogram.
    
    Args:
        vol (numpy.ndarray): The tomogram to show the segmentation on.
        seg (numpy.ndarray): The segmentation to show on the tomogram.
    """

    # Get the number of slices for the slider range
    max_slices = vol.shape[0] - 1
    middle_slice = int(max_slices // 2)

    # Launch the Interactive Widget
    interact(
        show_tomo_segmentation,
        tomo=fixed(vol), seg=fixed(seg),
        vol_slice=IntSlider(min=0, max=max_slices, step=1, value=middle_slice)
    )

# Define the plotting function
def show_tomo_segmentation(tomo, seg, vol_slice):
    """
    Show Segmentation on a Tomogram Slice.

    Args:
        tomo (numpy.ndarray): The tomogram to show the segmentation on.
        seg (numpy.ndarray): The segmentation to show on the tomogram.
        vol_slice (int): The slice index to show.
    """
    
    plt.figure(figsize=(20, 10))
    
    # Tomogram
    plt.subplot(1, 3, 1)
    plt.title('Tomogram')
    plt.imshow(tomo[vol_slice], cmap='gray')
    plt.axis('off')
    
    # Painted Segmentation
    plt.subplot(1, 3, 2)
    plt.title('Painted Segmentation from Picks')
    plt.imshow(seg[vol_slice], cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(tomo[vol_slice], cmap='gray')
    plt.imshow(seg[vol_slice], cmap='viridis', alpha=0.5)  # Add alpha=0.5 for 50% transparency
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_labeled_tomo_segmentation(tomo, seg, seg_labels, unique_values, vol_slice):

        # # Check unique values in segmentation to ensure correct mapping
        # unique_values = np.unique(seg)

    plt.figure(figsize=(20, 10))
    num_classes = len(seg_labels)        

    # Dynamically update the labels and colormap based on unique values
    seg_labels_filtered = {k: v for k, v in seg_labels.items() if k in unique_values}
    num_classes = len(seg_labels_filtered)

    # Create a discrete colormap
    colors = plt.cm.tab20b(np.linspace(0, 1, num_classes))  # You can use other colormaps like 'Set3', 'tab20', etc.
    cmap = mcolors.ListedColormap(colors)
    bounds = list(seg_labels_filtered.keys()) + [max(seg_labels_filtered.keys())]
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Tomogram plot
    plt.subplot(1, 2, 1)
    plt.title('Tomogram')
    plt.imshow(tomo[vol_slice], cmap='gray')
    plt.axis('off')

    # Prediction segmentation plot
    plt.subplot(1, 2, 2)
    plt.title('Prediction Segmentation')
    im = plt.imshow(seg[vol_slice], cmap=cmap)  # Use norm and cmap for segmentation
    plt.axis('off')

    # Add the labeled color bar
    cbar = plt.colorbar(im, ticks=list(seg_labels_filtered.keys()))
    cbar.ax.set_yticklabels([seg_labels_filtered[i] for i in seg_labels_filtered.keys()])  # Set custom labels

    plt.tight_layout()
    plt.show()    

def interact_points(
    tomo, config, run_id, user_id='octopi', 
    session_id = None, pt_size = 15,
    slice_proximity_threshold = 3
    ):
    """
    Interactively show the points on a tomogram.

    Args:
        tomo (numpy.ndarray): The tomogram to show the points on.
        run_id (str): The ID of the run to show the points on.
        user_id (str): The ID of the user to show the points on.
        session_id (str): The ID of the session to show the points on.
        slice_proximity_threshold (int): The threshold for the proximity of the points to the slice.
        pt_size (int): The size of the points to show.
    """

    # Load Copick Project and Run
    root = copick.from_file(config)
    run = root.get_run(run_id)

    # Get objects that can be Picked
    objects = [(obj.name, obj.label, obj.radius) for obj in root.pickable_objects if obj.is_particle]

    # Get the number of slices for the slider range
    max_slices = tomo.shape[0] - 1
    middle_slice = int(max_slices // 2)

    # Launch the Interactive Widget
    interact(
        show_tomo_points,
        tomo=fixed(tomo), run=fixed(run), objects=fixed(objects), 
        user_id=fixed(user_id), session_id=fixed(session_id), 
        slice_proximity_threshold=fixed(slice_proximity_threshold),
        pt_size=fixed(pt_size),
        vol_slice=IntSlider(min=0, max=max_slices, step=1, value=middle_slice)
    )

def show_tomo_points(
        tomo, run, objects, user_id, 
        vol_slice, session_id = None, 
        slice_proximity_threshold = 3, pt_size = 15
    ):
    """
    Show Coordinates on a Tomogram Slice.

    Args:
        tomo (numpy.ndarray): The tomogram to show the points on.
        run (copick.Run): The Copick Run to show the points from.
        objects (list): List of pickable objects.
        user_id (str): The ID of the user to show the points from.
        vol_slice (int): The slice index to show.
        session_id (str): The ID of the session to show the points from.
        slice_proximity_threshold (int): The threshold for the proximity of the points to the slice
    """


    plt.figure(figsize=(20, 10))

    plt.imshow(tomo[vol_slice],cmap='gray')
    plt.axis('off')

    for name,_,_ in objects:
        try:    
            coordinates = readers.coordinates(run, name=name, user_id=user_id, session_id=session_id)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=pt_size)
        except:
            pass

    plt.show()

def compare_tomo_points(tomo, run, objects, vol_slice, user_id1, user_id2, 
                        session_id1 = None, session_id2 = None, slice_proximity_threshold = 3):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(tomo[vol_slice],cmap='gray')
    plt.title(f'{user_id1} Picks')

    for name,_,_ in objects:
        try:
            coordinates = readers.coordinates(run, name=name, user_id=user_id1, session_id=session_id1)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=15)    
        except: 
            pass

    plt.subplot(1, 2, 2)
    plt.imshow(tomo[vol_slice],cmap='gray')
    plt.title(f'{user_id2} Picks')
    
    for name,_,_ in objects:
        try:
            coordinates = readers.coordinates(run, name=name, user_id=user_id2, session_id=session_id2)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=15)
        except:
            pass

    plt.axis('off')
    plt.show()

def plot_training_results(
    results, 
    class_names: Optional[List[str]] = None,
    save_plot: str = None,
    fig = None, axs = None):
    """
    Plot Training Results including Loss, Recall, Precision, and F1 Score.

    Args:
        results (dict): A dictionary containing training metrics.
        class_names (list, optional): List of class names for labeling. Defaults to None.
        save_plot (str): Path to save the plot image.
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on. Defaults to None.
        axs (numpy.ndarray, optional): Existing axes to plot on. Defaults to None.
    """

    # Create a 2x2 subplot layout
    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    else:
        # Clear previos plots
        for ax in axs.flatten():
            ax.clear()

    fig.suptitle("Metrics Over Epochs", fontsize=16)

    # Unpack the data for loss (logged every epoch)
    epochs_loss = [epoch for epoch, _ in results['loss']]
    loss = [value for _, value in results['loss']]
    val_epochs_loss = [epoch for epoch, _ in results['val_loss']]
    val_loss = [value for _,value in results['val_loss']]

    # Plot Training Loss in the top-left
    axs[0, 0].plot(epochs_loss, loss, label="Training Loss")
    axs[0, 0].plot(val_epochs_loss, val_loss, label='Validation Loss')
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].legend()
    axs[0, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

    # For metrics that are logged every `val_interval` epochs
    epochs_metrics = [epoch for epoch, _ in results['avg_recall']]
    
    # Determine the number of classes and names
    num_classes = len([key for key in results.keys() if key.startswith('recall_class')])

    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class {i+1}" for i in range(num_classes)]

    # Plot Recall in the top-right
    for class_idx in range(num_classes):
        recall_class = [value for _, value in results[f'recall_class{class_idx+1}']]
        axs[0, 1].plot(epochs_metrics, recall_class, label=f"{class_names[class_idx]}")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Recall")
    axs[0, 1].set_title("Recall per Class")
    # axs[0, 1].legend()
    axs[0, 1].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

    # Plot Precision in the bottom-left
    for class_idx in range(num_classes):
        precision_class = [value for _, value in results[f'precision_class{class_idx+1}']]
        axs[1, 0].plot(epochs_metrics, precision_class, label=f"{class_names[class_idx]}")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].set_title("Precision per Class")
    axs[1, 0].legend()
    axs[1, 0].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

    # Plot F1 Score in the bottom-right
    for class_idx in range(num_classes):
        f1_class = [value for _, value in results[f'f1_class{class_idx+1}']]
        axs[1, 1].plot(epochs_metrics, f1_class, label=f"{class_names[class_idx]}")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("F1 Score")
    axs[1, 1].set_title("F1 Score per Class")
    # axs[1, 1].legend()
    axs[1, 1].tick_params(axis='both', direction='in', top=True, right=True, length=6, width=1)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

    fig.savefig(save_plot)
    fig.canvas.draw()

    return fig, axs