# Installation Guide

## Quick Installation (Recommended)

octopi is available on PyPI and can be installed using pip:

```bash
pip install octopi
```
This will install the latest stable release along with all required dependencies.

## Development Installation

If you want to contribute to octopi or need the latest development version, you can install from source:

```bash
git clone https://github.com/chanzuckerberg/octopi.git
cd octopi
pip install -e .
```

The editable (-e) install ensures that local code changes are immediately reflected without reinstalling.

---

!!! success "Verify installation"

    After installation, verify that the command-line interface is available:

    ```bash
    octopi
    ```
    
    You should see output similar to the following:

    ```bash
    Octopi 🐙: 🛠️ Tools for Finding Proteins in 🧊 cryo-ET data                        
                                                                                    
    ╭─ Options ───────────────────────────────────────────────────────────────────────╮
    │ --help  -h  Show this message and exit.                                         │
    ╰─────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Pre-Processing ────────────────────────────────────────────────────────────────╮
    │ download        Download and (optionally) downsample tomograms from the         │
    │                 CryoET-DataPortal.                                              │
    │ import          Import MRC tomograms from a folder into a copick project.       │
    │ create-targets  Generate segmentation targets from CoPick configurations.       │
    ╰─────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Training ──────────────────────────────────────────────────────────────────────╮
    │ train           Train 3D CNN U-Net models for Cryo-ET semantic segmentation.    │
    │ model-explore   Perform model architecture search with Optuna.                  │
    ╰─────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Inference ─────────────────────────────────────────────────────────────────────╮
    │ segment           Segment volumes using trained neural network models.          │
    │ localize          Convert Segmentation Masks to 3D Particle Coordinates.        │
    │ membrane-extract  Extract membrane-bound picks based on proximity to organelle  │
    │                   or membrane segmentation.                                     │
    │ evaluate          Evaluate particle localization performance against ground     │
    │                   truth annotations.                                            │
    ╰─────────────────────────────────────────────────────────────────────────────────╯
    ```

## Next Steps

- [Import Your Data](data-import.md) - Learn how to import your tomograms into a copick project.
- [Quick Start Guide](quickstart.md) - Run your first particle picking experiment. 
- [Learn the API](../user-guide/api-tutorial.md) - Integrate Octopi into your Python workflows. 
