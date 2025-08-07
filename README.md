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

Octopi is availableon PyPI and can be installed using pip:
```bash
pip install octopi
```

⚠️ **Note**: One of the current dependencies is currently not working with pip 25. To temporarily reduce the pip version, run:
```bash
pip install --upgrade "pip<25"
```

### Basic Usage

octopi provides two main command-line interfaces:

```bash
# Main CLI for training, inference, and data processing
octopi --help

# HPC-specific CLI for submitting jobs to SLURM clusters
octopi-slurm --help
```

## 📚 Documentation

For detailed documentation, tutorials, CLI and API reference, visit our [documentation](https://chanzuckerberg.github.io/octopi/).

## 🤝 Contributing

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. 
Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.


