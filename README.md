# OCTOPI 🐙🐙🐙

[![License](https://img.shields.io/pypi/l/octopi.svg?color=green)](https://github.com/chanzuckerberg/octopi/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/octopi.svg?color=green)](https://pypi.org/project/octopi)
[![Python Version](https://img.shields.io/pypi/pyversions/octopi.svg?color=green)](https://www.python.org/)

**O**bject dete**CT**ion **O**f **P**rote**I**ns. A deep learning framework for Cryo-ET 3D particle picking with autonomous model exploration capabilities.

## 🚀 Introduction

octopi addresses a critical bottleneck in cryo-electron tomography (cryo-ET) research: the efficient identification and extraction of proteins within complex cellular environments. As advances in cryo-ET enable the collection of thousands of tomograms, the need for automated, accurate particle picking has become increasingly urgent.

Our deep learning-based pipeline streamlines the training and execution of 3D autoencoder models specifically designed for cryo-ET particle picking. Built on [copick](https://github.com/copick/copick), a storage-agnostic API, octopi seamlessly accesses tomograms and segmentations across local and remote environments. 

## 🧩 Core Features

- **3D U-Net Training**: Train and evaluate custom 3D U-Net models for particle segmentation
- **Automatic Architecture Search**: Explore optimal model configurations using Bayesian optimization via Optuna
- **Flexible Data Access**: Seamlessly work with tomograms from local storage or remote data portals
- **HPC Ready**: Built-in support for SLURM-based clusters
- **Experiment Tracking**: Integrated MLflow support for monitoring training and optimization
- **Dual Interface**: Use via command-line or Python API

## 🚀 Quick Start

### Installation

```bash
pip install octopi
```

### Basic Usage

octopi provides two main command-line interfaces:

```bash
# Main CLI for training, inference, and data processing
octopi --help
```

The main `octopi` command provides subcommands for:
- Data import and preprocessing
- Training label preparation
- Model training and exploration
- Inference and particle localization

```bash
# HPC-specific CLI for submitting jobs to SLURM clusters
octopi-slurm --help
```

The `octopi-slurm` command provides utilities for:
- Submitting training jobs to SLURM clusters
- Managing distributed inference tasks
- Handling batch processing on HPC systems

## 📚 Documentation

For detailed documentation, tutorials, CLI and API reference, visit our [documentation](https://chanzuckerberg.github.io/octopi/).

## 🤝 Contributing

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.


