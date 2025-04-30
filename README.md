# OCTOPUS 🐙
**O**bject dete**CT**ion of **P**roteins **U**sing **S**egmentation. A deep learning framework for Cryo-ET 3D particle picking with autonomous model exploration capabilities.

## 🚀 Introduction

OCTOPUS addresses a critical bottleneck in cryo-electron tomography (cryo-ET) research: the efficient identification and extraction of proteins within complex cellular environments. As advances in cryo-ET enable the collection of thousands of tomograms, the need for automated, accurate particle picking has become increasingly urgent.

Our deep learning-based pipeline streamlines the training and execution of 3D autoencoder models specifically designed for cryo-ET particle picking. Built on [copick](https://github.com/copick/copick), a storage-agnostic API, octopus seamlessly accesses tomograms and segmentations across local and remote environments. 

OCTOPUS offers a modular, deep learning-driven pipeline for:
*	Training and evaluating custom 3D U-Net models for particle segmentation.
*	Automatically exploring model architectures using Bayesian optimization via Optuna.
*	Performing inference for both semantic segmentation and particle localization.

Octopus empowers researchers to navigate the dense, intricate landscapes of cryo-ET datasets with unprecedented precision and efficiency without manual trial and error.

## Features

## Getting Started
### Installation and setup the environment
Inside the directory, run `pip install -e .` (eventually this package will be available on PyPI).  

To use CZI cloud MLflow tracker, add a `.env` in the root directory like below. You can get a CZI MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token) (note that a new token will be generated everytime you open this site).
```
MLFLOW_TRACKING_USERNAME = <Your_CZ_email>
MLFLOW_TRACKING_PASSWORD = <Your_mlflow_access_token>
```

## Usage

### 📁 Training Labels Preparation  

Use `create-targets` to create semantic masks for proteins of interest using annotation metadata. In this example lets generate picks segmentations for dataset 10439 from the CZ cryoET Dataportal (only need to run this step once). 
```
create-targets \
    --config config.json \
    --target apoferritin --target beta-galactosidase,slabpick,1 \
    --target ribosome,pytom,0 --target virus-like-particle,pytom,0 \
    --seg-target membrane \
    --tomogram-algorithm wbp --voxel-size 10 \
    --target-session-id 1 --target-segmentation-name remotetargets \
    --target-user-id train-octopus
```

### 🧠 Training a single 3D U-Net model  
```
Train a 3D U-Net model on the prepared datasets using the prepared target segmentations. We can use tomograms derived from multiple copick projects.  
train-model \
    --config experiment,config1.json \
    --config simulation,config2.json \
    --voxel-size 10 --tomo-algorithm wbp --Nclass 8 \
    --tomo-batch-size 50 --num-epochs 100 --val-interval 10 \
    --target-info remotetargets,train-octopus,1
```
Outputs will include model weights (.pth), logs, and training metrics.

### 🔍 Model exploration with Optuna

OCTOPUS🐙 supports automatic neural architecture search using Optuna, enabling efficient discovery of optimal 3D U-Net configurations through Bayesian optimization. This allows users to maximize segmentation accuracy without manual tuning.

To launch a model exploration job:
```
model-explore \
    --config experiment,/mnt/dataportal/ml_challenge/config.json \
    --config simulation,/mnt/dataportal/synthetic_ml_challenge/config.json \
    --voxel-size 10 --tomo-algorithm wbp --Nclass 8 \
    --model-save-path train_results
```
Each trial evaluates a different architecture and logs:
	•	Segmentation performance metrics
	•	Model weights and configs
	•	Training curves and validation loss

🔬 Trials are automatically tracked with MLflow and saved under the specified `--model-save-path`.

#### Optuna Dashboard

(TODO)
 
#### 📊 MLflow experiment tracking   

OCTOPUS supports MLflow for logging and visualizing model training and hyperparameter search results, including:
	•	Training loss/validation metrics over time
	•	Model hyperparameters and architecture details
	•	Trial comparison (e.g., best performing model)

You can use either a local MLflow instance, a remote (HPC) instance, or the CZI cloud server:

#### 🧪 Local MLflow Dashboard

To inspect results locally: `mlflow ui` and open http://localhost:5000 in your browser.

#### 🖥️ HPC Cluster MLflow Access (Remote via SSH tunnel)

If running OCTOPUS on a remote cluster (e.g., Biohub Bruno), forward the MLflow port. 
On your local machine: 
 `ssh -L 5000:localhost:5000 remote_username@remote_host` (in the case of Bruno the remote would be `login01.czbiohub.org`). 
 
Then on the remote terminal (login node): ` mlflow ui --host 0.0.0.0 --port 5000` to launch the MLFlow dashboard on a local borwser.

#### ☁️ CZI coreweave cluser

For the CZI coreweave cluser, MLflow is already hosted. Go to the CZI [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). 

🔐 A .env file is required to authenticate (see Getting Started section).
📁 Be sure to register your project name in MLflow before launching runs.

### 🔮 Segmentation
```
Generate segmentation prediction masks for tomograms in a given copick project.
inference \
    --config config.json \
    --seg-info predict,unet,1 \
    --model-config train_results/best_model_config.yaml \
    --model-weights train_results/best_model.pth \
    --voxel-size 10 --tomo-algorithm wbp --tomo-batch-size 25
```
Output masks will be saved to the corresponding copick project under the `seg-info` input.

### 📍 Localization
```
Convert the segmentation masks into particle coordinates. 
localize \
    --config config.json \
    --pick-session-id 1 --pick-user-id unet \
    --seg-info predict,unet,1
```

## Contact

email: [jonathan.schwartz@czii.org](jonathan.schwartz@czii.org)



