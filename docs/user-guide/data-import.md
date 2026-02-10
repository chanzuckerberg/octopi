# Importing Tomograms to Copick

octopi leverages [copick](https://github.com/copick/copick) to provide a flexible and unified interface for accessing tomographic data, whether it's stored locally or remotely on a HPC server or on our [CryoET Data Portal](https://cryoetdataportal.czscience.com). 

This page covers:

- How copick configuration files define a project
- Creating configurations for local filesystems and the Data Portal
- Importing volumes via the CLI or Python API

If you’re new to copick itself, the following upstream tutorials are excellent references:

- [Copick Quickstart](https://copick.github.io/copick/quickstart/) — Basic configuration and setup
- [Data Portal Tutorial](https://copick.github.io/copick/examples/tutorials/data_portal/) — Working with CryoET Data Portal data

## Configuration File

A **copick configuration file** defines how data are discovered, stored, and annotated within a project. In cryo-ET, this typically corresponds to a collection of tilt-series and their reconstructed tomograms.

Each configuration specifies:

- **Pickable objects** — proteins or structures that can be annotated or predicted
- **Static storage** — read-only data such as raw volumes and reference annotations
- **Overlay storage** — writable data such as predictions, labels, and derived results

The static path is never modified, while the overlay path is where octopi and copick write outputs through either the CLI or Python API.

??? example "Copick Config File (`config.json`)"

    The configuration file points copick to the project storage and defines the objects that octopi will segment or localize.

    ```json
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

        // Change this path to the location of sample_project
        "overlay_root": "local:///PATH/TO/EXTRACTED/PROJECT/",
        "overlay_fs_args": {
            "auto_mkdir": true
        }
    }
    ```

Copick provides a CLI for generating configuration files that mount either local filesystems or remote data sources such as the CryoET Data Portal.

=== "Local File System"

    Use this mode when your tomograms are already available on disk (e.g. HPC scratch or shared storage).

    ```bash
    copick config filesystem \
        --config config.json
        --overlay-root /mnt/24sep24a/run002 \
        --objects membrane,False \
        --objects apoferritin,True,60,4V1W \
        --proj-name 24sep24a \
        --proj-description "Synaptic Vesicles collected on 24sep24"
    ```

    We can define either objects that are continuous segmentations (e.g., organelles or memebranes) or coordinates. For pickable objects, we can store meta-data including the particle radius and a corresponding PDB-ID:
        
        - `--objects name,is_particle,radius,pdb_id`.

    ??? info "`copick config filesystem` parameters"

        | Parameter | Description |
        |----------|-------------|
        | `--overlay-root` | Writable overlay directory for predictions and annotations |
        | `--objects` | Object definition: `name,is_particle[,radius,pdb_id]` (repeatable) |
        | `--config` | Output path for the generated config file |
        | `--proj-name` | Human-readable project name |
        | `--proj-description` | Description stored in config metadata |

=== "Data Portal"

    When working with portal-hosted datasets, copick automatically populates pickable objects based on known annotations. Only the dataset ID and overlay location are required.

    ```bash
    copick config dataportal \
        -ds 10403 --overlay /mnt/10403/overlay \
        --output /mnt/10403/config.json
    ```

    ??? info "`copick config dataportal` parameters"

        | Parameter | Description |
        |----------|-------------|
        | `--dataset-id, -ds` | CryoET Data Portal dataset ID |
        | `--overlay` | Local overlay directory for writable outputs |
        | `--output` | Path to save the generated configuration file |

## Importing Volumes

### Recommended Resolution

For most workflows, we recommend working at a voxel size of **10 Å (1 nm)**. Higher-resolution tomograms can be downsampled during import to reduce memory usage and improve training and inference performance.

If you have tomograms stored locally in `*.mrc` format (e.g., from Warp, IMOD, or AreTomo), you can import them into a copick project using either copick directly or via octopi, which adds optional resampling and convenience features.

=== "Copick Import CLI"

    Use copick when volumes are already organized and no resampling is required.

    ```bash 
    copick add tomogram \
        -c /path/to/config.json \
        --tomo-type sart \
    ```

    ??? info "`copick add tomogram` parameters"

        | Parameter | Description |
        |----------|-------------|
        | `PATH` | Path to a tomogram file (`.mrc` or `.zarr`) or glob pattern. |
        | `--config, -c` | Path to the configuration file (or `COPICK_CONFIG`). |
        | `--run` | Name of the run; defaults to filename. |
        | `--run-regex` | Regex for extracting run names from filenames. |
        | `--tomo-type` | Logical tomogram type (e.g. `wbp`, `sart`). |
        | `--file-type` | Explicit file type (`mrc` or `zarr`). |
        | `--voxel-size` | Override voxel size stored in the header (Å). |
        | `--create-pyramid` | Build multiscale pyramid for visualization. |
        | `--pyramid-levels` | Number of pyramid levels. |
        | `--chunk-size` | Chunk size for Zarr storage. |
        | `--overwrite` | Overwrite existing tomograms. |
        | `--create` | Create tomogram entry if missing. |

=== "Octopi Import CLI"

    Octopi extends the import process with optional voxel-size conversion.

    ```bash
    octopi import \
        -p /path/to/mrc/files \
        -c /path/to/config.json \
        -alg denoised \
        --input-voxel-size 5 \
        --output-voxel-size 10
    ```

    When downsampling is unnecessary, simply omit the `--output-voxel-size` argument.

    ??? info "`octopi import` parameters"

        | Parameter | Description | Notes |
        |----------|-------------|-------|
        | `--path, -p` | Directory containing MRC tomograms | Required |
        | `--config, -c` | Copick configuration file | Alternative to datasetID |
        | `--tomo-alg, -alg` | Tomogram type name | e.g. `denoised` |
        | `--input-voxel-size, -ivs` | Input voxel size (Å) | Default: 10 |
        | `--output-voxel-size, -ovs` | Target voxel size (Å) | Optional |


=== "Import with the API"

    For non-standard layouts or programmatic workflows, volumes can be added directly through the Python API.

    ```python
    from copick.utils.io import writers
    import copick, mrcfile

    # Open Copick Project
    root = copick.from_file('/path/to/config.json')

    # Load the Volume
    vol = mrcfile.read('path/to/volume.mrc')

    # Write a New Run if not present
    run = copick.get_run('Run001')
    if run is None: run = copick.new-run('Run001')

    # Write the Volume into the Run
    voxel_size = 10
    tomo_alg = 'wbp'
    writers.tomogram(run, vol,voxel_size,tomo_alg)
    ```

## Downloading from the CryoET Data-Portal

The [CryoET Data Portal](https://cryoetdataportal.czscience.com) provides access to thousands of annotated tomograms. Octopi can work with this data in two ways:

### 1. Direct Portal Access

You can train models directly using data from the portal without downloading:

```bash
octopi train-model \
    --config portal_config.json \
    --datasetID 10445 \
    --voxel-size 10
```

### 2. Local Download and Processing

For larger datasets or when running multiple experiments, it is recommended to download the data first:

```bash
octopi download \
    -c /path/to/config.json \
    --datasetID 10445 \
    --overlay-path /path/to/saved/zarrs \
    --input-voxel-size 5 --output-voxel-size 10 \
    --target-type wbp --source-type wbp-denoised-denoiset-ctfdeconv 
```

??? info "`octopi download` parameters"

    | Parameter | Description |
    |----------|-------------|
    | `--config, -c` | Existing copick configuration file |
    | `--datasetID, -ds` | CryoET Data Portal dataset ID |
    | `--overlay-path` | Overlay directory when creating a new project |
    | `--input-voxel-size` | Original voxel size of portal tomograms (Å) |
    | `--output-voxel-size` | Target voxel size after downsampling (Å) |
    | `--target-type` | Local tomogram type name in copick |
    | `--source-type` | Portal tomogram type label |

Similar to local MRC import, you can downsample portal data by specifying both `--input-voxel-size` and `--output-voxel-size` parameters.  To find available tomogram names for a dataset available on the portal, use:

```bash
copick browse -ds <datasetID>
```

This will save these tomograms locally under the `--target-tomo-type` flag.

## Next Steps

Once your data is imported, you can:

- [Try the Quick Start](quickstart.md) - Complete end-to-end workflow example
- [Prepare Training Data](../user-guide/labels.md) - Set up your particle annotations
- [Start Training Models](../user-guide/training.md) - Train custom 3D U-Net models
- [Run Inference](../user-guide/inference.md) - Apply trained models to new data