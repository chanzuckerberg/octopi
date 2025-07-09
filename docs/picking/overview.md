# Manual Particle Picking

## Overview

The ChimeraX-Copick plugin provides an intuitive interface for manual particle picking in cryo-electron tomograms. This tool bridges the gap between automated processing pipelines and the precision of human annotation, allowing researchers to:

- **Interactively pick particles** in 3D tomographic volumes
- **Validate and refine** automated picking results
- **Annotate multiple object types** (particles, membranes, gold fiducials)
- **Work with both local and remote datasets** seamlessly
- **Collaborate** through shared picking sessions

The plugin integrates directly with ChimeraX's powerful 3D visualization capabilities, providing an efficient workflow for particle coordinate generation. Whether you're working with data stored locally or accessing remote repositories, the plugin offers a consistent interface for high-quality manual picking.

## Key Features

### 🖼️ Gallery View
Browse and select from multiple tomograms in an organized thumbnail gallery, making it easy to navigate large datasets and track picking progress across multiple volumes.

### 🎯 Multi-Object Support
Pick different particle types with distinct colors and sizes. Each object type can be configured with specific visual properties and physical parameters to match your experimental needs.

### 📊 Real-Time Visualization
See picks overlaid directly on tomogram data with immediate visual feedback. The 3D environment allows for precise spatial positioning and context-aware picking.

### 🔄 Flexible Data Access
Support for both local filesystem and remote SSH connections means you can work with data regardless of its location - whether it's on your local machine or a remote cluster.

### 📋 Standardized Output
Generates picks in formats compatible with downstream processing pipelines, ensuring seamless integration with your existing workflow.

### 👥 Collaborative Picking
Multiple users can work on the same dataset with session management, allowing for distributed annotation efforts and quality control processes.


The following sections will guide you through each step of this workflow, from initial setup to advanced picking strategies.