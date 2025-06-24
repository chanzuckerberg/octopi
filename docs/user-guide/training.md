# Training Octopi Models

This guide covers everything you need to know about training 3D U-Net models with Octopi, from basic single-model training to advanced hyperparameter optimization with Bayesian methods.

## Overview

Octopi offers two main training approaches:

1. **Single Model Training** - Train a specific architecture with defined parameters
2. **Model Exploration** - Automatically discover optimal architectures using Bayesian optimization

For most use cases, we recommend starting with model exploration to find the best architecture for your data.

## Single Model Training

For specific use cases or when you have a known good architecture, you can train a single model directly.

### Basic Training Command

```bash
octopi train-model \
    --config config.json \
    --voxel-size 10 --tomo-alg wbp --Nclass 8 \
    --tomo-batch-size 50 --num-epochs 100 --val-interval 10 \
    --target-info targets,octopi,1
```

## Model Exploration with Bayesian Optimization

![Bayesian Optimization Workflow](../assets/bo_workflow.png)
*OCTOPI's automated architecture search uses Bayesian optimization to efficiently explore the space of possible 3D U-Net configurations, maximizing segmentation performance.*

### Why Use Model Exploration?

- **Automatic optimization** - No manual hyperparameter tuning required
- **Efficient search** - Bayesian methods are smarter than grid search
- **Optimal performance** - Finds architectures tailored to your specific data
- **Time savings** - Avoids trial-and-error experimentation

### Launch Model Exploration

```bash
octopi model-explore \
    --config experiment,/mnt/dataportal/ml_challenge/config.json \
    --config simulation,/mnt/dataportal/synthetic_ml_challenge/config.json \
    --voxel-size 10 --tomo-alg wbp --Nclass 8 \
    --model-save-path train_results
```
### What Model Exploration Optimizes

Each trial evaluates different architectural choices:
- **Network depth** - Number of encoder/decoder layers
- **Feature channels** - Width of convolutional layers  
- **Regularization** - Dropout and weight decay parameters

### Exploration Outputs

The exploration process generates:
- **Performance metrics** - F1 scores, precision, recall for each trial
- **Model configurations** - Architecture specifications for top performers
- **Training artifacts** - Weights and logs for promising models
- **Optimization history** - Complete trial progression and results

## Monitoring and Tracking

### Optuna Dashboard

Monitor your hyperparameter search progress in real-time using the Optuna dashboard.

**Setup Options:**
- **VS Code Extension** - Install the Optuna extension for integrated monitoring
- **CLI Dashboard** - Follow the [Optuna dashboard guide](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html)

The dashboard provides:
- **Trial progress** - Real-time optimization status
- **Parameter importance** - Which hyperparameters matter most
- **Optimization history** - Performance trends over trials
- **Best trial identification** - Top-performing configurations

### MLflow Experiment Tracking

Octopi integrates with MLflow for comprehensive experiment management and visualization.

**🧪 Local MLflow Dashboard**
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

**🖥️ HPC Cluster Access (Remote SSH Tunnel)**

For remote clusters like Biohub Bruno:

1. **Forward the port** (on your local machine):
   ```bash
   ssh -L 5000:localhost:5000 remote_username@login01.czbiohub.org
   ```

2. **Launch MLflow** (on the remote terminal):
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

#### What MLflow Tracks

MLflow automatically logs:
- **Training metrics** - Loss curves and validation performance over time
- **Model parameters** - Architecture details and hyperparameter values  
- **Trial comparisons** - Performance across different model configurations
- **Artifacts** - Model weights, configuration files, and training plots

## Next Steps

After successful training:

- **[Run Inference](inference.md)** - Apply your trained models to new tomograms and assess model quality on test data 