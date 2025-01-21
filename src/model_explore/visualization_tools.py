import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from model_explore import io
import numpy as np

# Define the plotting function
def show_tomo_segmentation(tomo, seg, vol_slice):
    
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


def show_tomo_points(tomo, run, objects, user_id, vol_slice, session_id = None, slice_proximity_threshold = 3):
    plt.figure(figsize=(20, 10))

    plt.imshow(tomo[vol_slice],cmap='gray')
    plt.axis('off')

    for name,_,_ in objects:
        try:    
            coordinates = io.get_copick_coordinates(run, name=name, user_id=user_id, session_id=session_id)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=15)
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
            coordinates = io.get_copick_coordinates(run, name=name, user_id=user_id1, session_id=session_id1)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=15)    
        except: 
            pass

    plt.subplot(1, 2, 2)
    plt.imshow(tomo[vol_slice],cmap='gray')
    plt.title(f'{user_id2} Picks')
    
    for name,_,_ in objects:
        try:
            coordinates = io.get_copick_coordinates(run, name=name, user_id=user_id2, session_id=session_id2)
            close_points = coordinates[np.abs(coordinates[:, 0] - vol_slice) <= slice_proximity_threshold]
            plt.scatter(close_points[:, 2], close_points[:, 1], label=name, s=15)
        except:
            pass

    plt.axis('off')
    plt.show()