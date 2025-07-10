# Copick Configuration File Setup

Before you can start manual particle picking, you need a copick configuration file that defines your project structure, data locations, and pickable objects. This guide will help you create the right config file for your setup. 

- **Note** This tutorial is an extension to the [Data-Import Tutorial](../getting-started/data-import.md).

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

<details markdown="1">
<summary><strong>🔍 Configuration Parameters Explained  </strong></summary> 

**`pickable_objects`**: Array of objects that can be picked

  - **`name`**: Identifier for the object
  - **`is_particle`**: Boolean indicating if it's a particle (true) or surface/membrane (false)
  - **`label`**: Numeric label for segmentation
  - **`color`**: RGBA color values [R, G, B, A] (0-255)
  - **`radius`**: Size of the object in Angstroms

**`static_root`**: Path to immutable data (tomograms, segmentations)

  - Use `ssh://` prefix for remote directories
  - Use `local://` prefix for local datasets

**`overlay_root`**: Path to mutable data (picks, annotations)

  - Use `ssh://` prefix for remote directories
  - Use `local://` prefix for local datasets

**`static_fs_args`** / **`overlay_fs_args`**: Connection parameters for remote access

  - **`host`**: Usually "localhost" when using SSH tunnel
  - **`port`**: Port number (2222 in our SSH tunnel example)

</details>

**Note** You can mix overlay and static roots - for example, having an overlay root pointing to a local file location and static root pointing to a remote directory (or vice versa).

## Generating Configuration Templates

Copick provides convenient commands to generate configuration templates for different setups:

**Local / Remote Filesystem Setup:**
```bash
copick config filesystem --help
```

**CryoET DataPortal Filesystem Setup:**
```bash
copick config dataportal --help
```

---

## Next Steps

[ChimeraX Plugin](chimerax-plugin.md) - Now that we have a config file available, let's get straight to picking! 