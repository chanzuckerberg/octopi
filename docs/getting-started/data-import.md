# Data Import Guide

octopi leverages [copick](https://github.com/copick/copick) to provide a flexible and unified interface for accessing tomographic data, whether it's stored locally or on the [CryoET Data Portal](https://cryoetdataportal.czscience.com). This guide explains how to work with both data sources.

## Data Resolution

Before importing data, it's important to consider the resolution. We recommend working with tomograms at a voxel size of 10 Å (1 nm) for optimal performance. This resolution:
- Reduces memory usage
- Speeds up training and inference
- Maintains sufficient detail for particle picking
- Provides a good balance between accuracy and computational efficiency

## Importing Local MRC Files

If you have tomograms stored locally in *.mrc format (e.g., from Warp, IMOD, or AreTomo), you can import them into a copick project:

```bash
octopi import-mrc-volumes \
    --input-folder /path/to/mrc/files \
    --config /path/to/config.json \
    --target-tomo-type denoised \
    --input-voxel-size 5 \
    --output-voxel-size 10
```

This command will:
1. Read the MRC files from your local directory
2. Create a copick project structure
3. Downsample the tomograms to 10 Å resolution
4. Store the processed data in a format ready for training

## Working with CryoET Data Portal

The CryoET Data Portal provides access to thousands of annotated tomograms. Octopi can work with this data in two ways:

### 1. Direct Portal Access

You can train models directly using data from the portal without downloading:

```bash
octopi train-model \
    --config portal_config.json \
    --datasetID 10445 \
    --voxel-size 10
```

This approach:
- Streams data directly from the portal
- Saves local storage space
- Ideal for initial experiments or small datasets
- Requires stable internet connection

### 2. Local Download and Processing

For larger datasets or when running multiple experiments, it is recommended to download the data first:

```bash
octopi download-dataportal \
    --config /path/to/config.json \
    --datasetID 10445 \
    --overlay-path path/to/saved/zarrs \
    --input-voxel-size 5 \
    --output-voxel-size 10 \
    --dataportal-name wbp
```

This approach:
- Downloads and processes data locally
- Enables faster training iterations
- Allows offline work
- Better for large-scale experiments

## Configuring Data Access

The copick configuration file (`config.json`) is the key to managing data access. Here's a basic example:

```json
{
    "project_name": "my_project",
    "data_sources": {
        "local": {
            "type": "local",
            "path": "/path/to/local/data"
        },
        "portal": {
            "type": "portal",
            "dataset_id": "10445",
            "voxel_size": 10
        }
    }
}
```

## Best Practices

1. **Resolution Management**:
   - Always specify both input and output voxel sizes
   - Use 10 Å as the target resolution
   - Consider your GPU memory when choosing batch sizes

2. **Data Organization**:
   - Keep local data in a well-organized directory structure
   - Use meaningful names for your copick projects
   - Document the source and processing steps

3. **Portal Usage**:
   - Download data for intensive training sessions
   - Use direct access for quick experiments
   - Check portal documentation for dataset details

4. **Storage Considerations**:
   - Monitor local storage when downloading portal data
   - Use appropriate compression for local storage
   - Consider using external storage for large datasets

## Next Steps

- Learn about [training label preparation](../user-guide/training.md)
- Explore [model training options](../user-guide/training.md)
- Check out [inference and localization](../user-guide/inference.md) 