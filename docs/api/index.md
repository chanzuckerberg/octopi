# API Overview

The octopi Python API gives you full programmatic control over the particle picking pipeline — useful for scripting large-scale workflows, integrating into notebooks, or customizing training beyond what the CLI exposes.

## Core Workflow

<div class="grid cards" markdown>

-   :octicons-rocket-24: **Quick Start**

    End-to-end pipeline in one script — target creation, training, segmentation, localization, and evaluation.

    [:octicons-arrow-right-24: Get started](quick-start.md)

-   :octicons-cpu-24: **Training**

    Set up data generators, configure model architectures, choose loss functions, and run model exploration.

    [:octicons-arrow-right-24: Train](training.md)

-   :octicons-play-24: **Inference**

    Segment tomograms, localize particle coordinates, and evaluate against ground truth.

    [:octicons-arrow-right-24: Run inference](inference.md)

-   :octicons-plus-circle-24: **Adding New Models**

    Implement a custom architecture and register it with the octopi model registry.

    [:octicons-arrow-right-24: Extend](adding-new-models.md)

</div>