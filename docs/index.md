# Octopi 🐙🐙🐙

<strong>O</strong>bject dete<strong>CT</strong>ion <strong>O</strong>f <strong>P</strong>rote<strong>I</strong>ns. A deep learning framework for automated 3D particle picking in cryo-electron tomography (cryo-ET).

![Octopi](assets/octopi.png)

---

## Why OCTOPI?

<div class="grid cards" markdown>

-   :material-molecule: **Built for cryo-ET**

    3D U-Net models designed specifically for the challenges of tomographic data — missing wedge, low SNR, and anisotropic resolution.

-   :material-magnify-scan: **Autonomous architecture search**

    Bayesian optimization via Optuna finds the best model architecture for your data — no manual hyperparameter tuning required.

-   :material-layers-triple: **Storage-agnostic data layer**

    Built on [CoPick](https://github.com/copick/copick) — seamlessly reads tomograms and writes picks from local disk, S3, or any remote store.

-   :material-server: **HPC-ready**

    Submit training and architecture search jobs directly to SLURM clusters. Multi-GPU inference included out of the box.

</div>

---

## Tutorials

### CLI

<div class="grid cards" markdown>

-   :octicons-database-24: **Import Data**

    Set up a CoPick project and import your tomograms and initial picks.

    [:octicons-arrow-right-24: Import data](user-guide/data-import.md)

-   :fontawesome-solid-crosshairs: **Pick Particles**

    Generate initial particle picks using the interactive GUI or existing pick files.

    [:octicons-arrow-right-24: Pick particles](user-guide/pick-particles.md)

-   :octicons-cpu-24: **Train Models**

    Train a 3D U-Net or run autonomous architecture search with `model-explore`.

    [:octicons-arrow-right-24: Train](user-guide/training.md)

-   :octicons-play-24: **Inference**

    Segment tomograms, localize particles, and evaluate against ground truth.

    [:octicons-arrow-right-24: Run inference](user-guide/inference.md)

</div>

### Python API

<div class="grid cards" markdown>

-   :octicons-code-24: **API Overview**

    Introduction to driving octopi programmatically from Python.

    [:octicons-arrow-right-24: Read more](api/index.md)

-   :octicons-rocket-24: **Quick Start**

    End-to-end particle picking in a Jupyter notebook.

    [:octicons-arrow-right-24: Get started](api/quick-start.md)

-   :octicons-book-24: **Training Guide**

    Customize training loops, loss functions, and data augmentation via the API.

    [:octicons-arrow-right-24: Customize](api/training.md)

-   :octicons-plus-circle-24: **Adding New Models**

    Register new MONAI architectures in the octopi model registry.

    [:octicons-arrow-right-24: Extend](api/adding-new-models.md)

</div>

---

## Getting Help

Open an issue on our [GitHub repository](https://github.com/chanzuckerberg/octopi).
