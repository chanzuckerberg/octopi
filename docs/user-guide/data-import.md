# Data Import Guide

octopi leverages [copick](https://github.com/copick/copick) to provide a flexible and unified interface for accessing tomographic data, whether it's stored locally or remotely on a HPC server or on our [CryoET Data Portal](https://cryoetdataportal.czscience.com). This guide explains how to work with both data sources. If you need help creating these configuration files, detailed tutorials are available:

## Data Resolution Recommendation

Before importing data, it's important to consider the resolution. We recommend working with tomograms at a voxel size of **10 Å (1 nm)** or larger for optimal performance in deep learning workflows. This resolution provides the best balance between:

- **Computational efficiency** - Manageable file sizes for training
- **Feature preservation** - Sufficient detail for accurate particle detection
- **Memory usage** - Fits within typical GPU memory constraints

You can downsample higher-resolution tomograms during import using the built-in downsampling options.

## Starting a New Copick Project

The copick configuration file points to a directory that stores all tomograms, coordinates, and segmentations in an overlay root. Generate a config file using the command line, you can define biological objects during project creation:

```bash
copick config filesystem \
    --overlay-root /path/to/overlay \
    --objects ribosome,True,130,6QZP \
    --objects apoferritin,True,65 \
    --objects membrane,False
```

### Understanding the `--objects` Flag

The `--objects` flag accepts 2-4 elements separated by commas:

1. **Particle name** (required): e.g., `ribosome`
2. **Is pickable** (required): `True` for particles, `False` for continuous segmentations
3. **Particle radius** (optional): in Ångströms, e.g., `130`
4. **PDB ID** (optional): reference structure, e.g., `6QZP`

This structure supports both particle picking for sub-tomogram averaging and broader 3D segmentation tasks. Octopi is designed to train models from copick projects for:

- Object 3D localization and particle picking
- Volumetric segmentation of cellular structures
- General 3D dataset annotation and analysis

<details markdown="1">
<summary><strong>💡 Example Copick Config File (config.json) </strong></summary>

The resulting `config.json` file would look like this: 

```bash
{
    "name": "test",
    "description": "A test project description.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "7P6Z",
            "label": 1,
            "color": [0, 255, 0, 255],
            "radius": 150,
            "map_threshold": 0.037

        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 2,
            "color": [0, 0, 0, 255]
        }
    ],

    "overlay_root": "local:///path/to/overlay",
    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "local:///path/to/static",
    "static_fs_args": {
        "auto_mkdir": true
    }   
}
```

**Directory Structure:**

- **Overlay root:** Writable directory where new results can be added, modified, or deleted
- **Static root:** Read-only directory that never gets manipulated (frozen data)

**Path Types:**

- **Local paths:** `local:///path/to/directory`
- **Remote paths:** `ssh://server/path/to/directory`

The `copick config filesystem` command assumes local paths, but you can edit the config file to specify remote locations.

</details>

## Starting a Data Portal Copick Project 

Create a copick project that automatically syncs with the [CryoET Data Portal](https://cryoetdataportal.czscience.com): 

```bash
copick config dataportal --dataset-id DATASET_ID --overlay-root /path/to/overlay
```

This command generates a config file that syncs data from the portal with local or remote repositories. You only need to specify the dataset ID and the overlay or static path - pickable objects will automatically be populated from the dataset.

**Benefits:**

- Automatically populates pickable objects from the dataset
- Seamless integration with portal data
- Combines remote portal data with local overlay storage

## Importing Local MRC Volumes

### Prerequisites

This workflow assumes:

- **All tomogram files are in a flat directory structure (single folder)**
- Files are in **MRC format** (`*.mrc`)

### Import Command

If you have tomograms stored locally in `*.mrc` format (e.g., from Warp, IMOD, or AreTomo), you can import them into a copick project:

```bash
copick add tomogram \
    --config config.json \
    --tomo-type denoised \
    --voxel-size 10 \
    --no-create-pyramid \
    'path/to/volumes/*.mrc'
```

<details markdown="1">
<summary><strong>💡 Import Parameters Description </strong></summary>

### Parameter Descriptions

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--config` | Path to copick config file | `config.json` |
| `--tomo-type` | Name for the tomogram type in your copick project | `denoised`, `wbp`, `raw` |
| `--voxel-size` | Voxel size of the tomograms (in Ångströms) | `10` (for 10Å data) |
| `--no-create-pyramid` | Skip pyramid generation for faster import | (flag, no value) |
| `'path/to/volumes/*.mrc'` | Path to your MRC file(s) - supports wildcards | `'data/*.mrc'` |

</details>

In the case the imported volumes are less than 10 Å, we can downsample then with Fourier cropping with the following CLI command (will be available soon).

## Downloading from the CryoET Data-Portal

The [CryoET Data Portal](https://cryoetdataportal.czscience.com) provides access to thousands of annotated tomograms. Octopi can work with this data in two ways:

### 1. Direct Portal Access

You can train models directly using data from the portal without downloading:

```bash
octopi train-model \
    --config portal_config.json \
    --voxel-size 10 --tomo-alg denoised \
    --trainRunIDs 17040,17055,17079,17098,10435 \
    --validateRunIDs 17149
```

**Recommendation:** Use 6-10 tomograms for training and validation (more if your target is sparse). Refer to the [training tutorial](../user-guide/training-basics.md) for more details.

To find available tomogram names for a dataset available on the portal, use:

```bash
copick browse -ds <datasetID>
```

### 2. Local Download and Processing (Recommended)

For larger datasets or when running multiple experiments, it is recommended to download the data first:

```bash
octopi download-dataportal \
    --config /path/to/config.json \
    --datasetID 10445 \
    --overlay-path /path/to/overlay \
    --input-voxel-size 8.627 --output-voxel-size 10 \
    --dataportal-name sirt-raw --target-tomo-type sirt
```
**Notes:**

- `--target-tomo-type` is the tomogram name that is saved locally to our project.
- The `--output-voxel-size` flag is optional, if this is ommit we'll simply save the tomograms at the original resolution.

## Advanced Import Options

If your data doesn't meet the standard requirements (flat directory structure + MRC format), please refer to our [API Import Documentation](../api/importing-volumes.md), which covers:

- Different file formats (TIFF, HDF5, etc.)
- Custom import scripts

## Next Steps

Once your data is imported, you can:

- [Try the Quick Start](quickstart.md) - Complete end-to-end workflow example
- [Prepare Training Data](../user-guide/labels.md) - Set up your particle annotations
- [Start Training Models](../user-guide/training.md) - Train custom 3D U-Net models
- [Run Inference](../user-guide/inference.md) - Apply trained models to new data