# Creating Training Targets

In this step, we will prepare the target data necessary for training our model and predicting the coordinates of proteins within a tomogram.

We will use Copick to manage the filesystem, extract runIDs, and create spherical targets corresponding to the locations of proteins. Key tasks include:

* Generating Targets: For each tomogram, we extract particle coordinates and generate spherical targets based on these coordinates, and save the target data in OME Zarr format.

* The target dimensions are determined with an associated tomogram, (specified by the `tomogram-algorithm` and `voxel-size` parameters).

* We can include continuous segmentations like organelles and membranes as additional targets with the `seg-target` flag. 

The segmentations are saved with the associated `target-session-id`, `target-user-id` and `target-name` flags.  

## Method 1: Automated Query

The simplest approach is to let Octopi automatically find all pickable objects from a specific annotation source.

#### Basic Command

```bash
octopi create-targets \
    --config config.json \
    --picks-user-id data-portal --picks-session-id 0 \
    --seg-target membrane \
    --tomo-alg wbp --voxel-size 10 \
    --target-session-id 1 --target-segmentation-name targets \
    --target-user-id octopi
```

## Method 2: Manual Specification

For more control, manually specify which protein types and annotation sources to include. Users can define a subset of pickable objects (from the CoPick configuration file) by specifying the name, and optionally the userID and sessionID. This allows for creating customizable training targets from varying submission sources. 

#### Advanced Command

```bash
octopi create-targets \
    --config config.json \
    --target apoferritin --target beta-galactosidase,slabpick,1 \
    --target ribosome,pytom,0 --target virus-like-particle,pytom,0 \
    --seg-target membrane \
    --tomo-alg wbp --voxel-size 10 \
    --target-session-id 1 --target-segmentation-name targets \
    --target-user-id octopi
```

### Check Target Quality

Refer to the notebook in `notebooks/inspect_segmentation_targets.ipynb` to see how to load a segmentation target and overlay it over the tomograms. Additional, use the ChimeraX Copick plug-in to also check the targets.