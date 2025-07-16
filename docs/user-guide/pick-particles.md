# ChimeraX-Copick Plugin Guide

## Installation

### 1. Download ChimeraX
First, download and install ChimeraX from the [official website](https://www.cgl.ucsf.edu/chimerax/)

### 2. Install the Copick Plugin
Once ChimeraX is installed:

1. Open ChimeraX
2. Go to **Tools → More Tools...** to open the toolshed
3. Search for `copick` in the search bar
4. Download and install the copick plugin

## Setup for Remote Data Access

### SSH Tunnel Configuration
If your data is stored on a remote repository (not locally available), you'll need to establish an SSH tunnel:

```bash
ssh -L 2222:localhost:22 login01.czbiohub.org
```

This command creates a local port forwarding from port 2222 to the remote server's port 22, allowing secure access to remote data.

**Note:** If you're working with data that's already available on your local machine, you can skip this SSH tunnel step.

## Configuration File Setup

### Config File Structure
Create a JSON configuration file that defines your project parameters and data locations. The config file should specify:

- **Pickable objects**: All objects that can be picked in your project
- **Data roots**: Paths to static and overlay data
- **Connection parameters**: For remote access

#### Example Remote Data Configuration
```json
{
    "config_type": "filesystem",
    "name": "lysosome",
    "description": "Picking vATPase on lysosomes.",
    "version": "0.5.0",
    "pickable_objects": [
        {
            ...
        }        
    ],
    "static_root": "ssh:///hpc/projects/group.czii/krios1.processing/copick/24sep20a/run002/static",
    "static_fs_args": {        
        "host": "localhost",
        "port": 2222
    },
    "overlay_root": "ssh:///hpc/projects/group.czii/krios1.processing/copick/24sep20a/run002/overlay",
    "overlay_fs_args": {
        "host": "localhost",
        "port": 2222
    }
}
```

#### Example Local Data Configuration
```json
{
    "config_type": "filesystem",
    "name": "lysosome_local",
    "description": "Picking vATPase on lysosomes - local data.",
    "version": "0.5.0",
    "pickable_objects": [
        {
            ...
        }        
    ],
    "static_root": "local:///path/to/local/data/static",
    "overlay_root": "local:///path/to/local/data/overlay"
}
```

### Configuration Parameters Explained

- **`pickable_objects`**: Array of objects that can be picked
  - **`name`**: Identifier for the object
  - **`is_particle`**: Boolean indicating if it's a particle (true) or surface/membrane (false)
  - **`label`**: Numeric label for segmentation
  - **`color`**: RGBA color values [R, G, B, A] (0-255)
  - **`radius`**: Size of the object in Angstroms

- **`static_root`**: Path to immutable data (tomograms, segmentations)
  - Use `ssh://` prefix for remote directories
  - Use `local://` prefix for local datasets

- **`overlay_root`**: Path to mutable data (picks, annotations)
  - Use `ssh://` prefix for remote directories
  - Use `local://` prefix for local datasets

- **`static_fs_args`** / **`overlay_fs_args`**: Connection parameters for remote access
  - **`host`**: Usually "localhost" when using SSH tunnel
  - **`port`**: Port number (2222 in our SSH tunnel example)

## Launching the Plugin

### Starting Copick
1. Open ChimeraX
2. In the command line, enter:
   ```
   copick start /path/to/your/config.json
   ```
3. The plugin will launch and display a gallery of available tomograms

### Workflow Overview
1. **Select a tomogram**: Click on any thumbnail in the gallery to open the tomogram
2. **Navigate the data**: Use ChimeraX's navigation tools to explore the 3D volume
3. **Pick particles**: Select the appropriate object type and begin picking
4. **Save progress**: Your picks are automatically saved to the overlay directory

## Tips for Effective Picking

- **Use appropriate zoom levels**: Zoom in for precise picking, zoom out for context
- **Leverage different views**: Use orthogonal slices and 3D views for better spatial understanding
- **Color coding**: Different object types have distinct colors as defined in the config
- **Batch processing**: Work through multiple tomograms systematically using the gallery view

## Troubleshooting

### Common Issues
- **SSH connection problems**: Ensure the SSH tunnel is active and the port is correct
- **Path errors**: Verify that the paths in your config file are correct and accessible
- **Permission issues**: Check that you have read/write access to the specified directories

### Verification Steps
1. Test SSH connection before launching ChimeraX
2. Verify config file syntax using a JSON validator
3. Check that both static and overlay directories exist and are accessible