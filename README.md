# OCTOPUS 🐙
**O**bject dete**CT**ion of **P**roteins **U**sing **S**egmentation. A deep learning framework for Cryo-ET 3D particle picking with autonomous model exploration capabilities.

## Introduction

OCTOPUS addresses a critical bottleneck in cryo-electron tomography (cryo-ET) research: the efficient identification and extraction of proteins within complex cellular environments. As advances in cryo-ET enable the collection of thousands of tomograms, the need for automated, accurate particle picking has become increasingly urgent.

Our deep learning-based pipeline streamlines the training and execution of 3D autoencoder models specifically designed for cryo-ET particle picking. Built on [copick](https://github.com/copick/copick), a storage-agnostic API, DeepFindET seamlessly accesses tomograms and segmentations across local and remote environments. 

A key feature of DeepFindET is its automatic model architecture search using Bayesian optimization, allowing researchers to discover optimal neural network architectures for their specific datasets without manual trial and error.

Integration with our [ChimeraX plugin](https://github.com/copick/chimerax-copick) enables researchers to easily annotate new tomograms and visualize segmentation results or particle coordinates.

DeepFindET empowers researchers to navigate the dense, intricate landscapes of cryo-ET datasets with unprecedented precision and efficiency.

## Getting Started
### Installation and setup the environment
Inside the directory, run `pip install -e .` (eventually this package will be available on PyPI).  

To use CZI cloud MLflow tracker, add a `.env` in the root directory like below. You can get a CZI MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token) (note that a new token will be generated everytime you open this site).
```
MLFLOW_TRACKING_USERNAME = <Your_CZ_email>
MLFLOW_TRACKING_PASSWORD = <Your_mlflow_access_token>
```

## Usage

### Prepare the dataset 
Generate picks segmentations for dataset 10439 from the CZ cryoET Dataportal (only need to run this step once). 
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

### Training a single 3D U-Net model  
```
Train a 3D U-Net model on the prepared datasets using the prepared target segmentations. We can use tomograms derived from multiple copick projects.  
train-model \
    --config experiment,config1.json \
    --config simulation,config2.json \
    --voxel-size 10 --tomo-algorithm wbp --Nclass 8 \
    --tomo-batch-size 50 --num-epochs 100 --val-interval 10 \
    --target-info remotetargets,train-octopus,1
```

### Model hyperparameter exploration with Optuna
```
Automatically search for the best model architecture and hyperparameters using Bayesian optimization to improve segmentation performance. 
model-explore \
    --config experiment,/mnt/dataportal/ml_challenge/config.json \
    --config simulation,/mnt/dataportal/synthetic_ml_challenge/config.json \
    --voxel-size 10 --tomo-algorithm wbp --Nclass 8 \
    --model-save-path train_results
```
 
## MLflow tracking   
To view the tracking results for model-explorations, go to the CZI [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). Note the project name needs to be registered first.


### Inference (Segmentation)
```
Generate segmentation prediction masks for tomograms in a given copick project.
inference \
    --config config.json \
    --seg-info predict,unet,1 \
    --model-config train_results/best_model_config.yaml \
    --model-weights train_results/best_model.pth \
    --voxel-size 10 --tomo-algorithm wbp --tomo-batch-size 25
```

### Inference (Localization)
```
Convert the segmentation masks into particle coordinates. 
localize \
    --config config.json \
    --pick-session-id 1 --pick-user-id unet \
    --seg-info predict,unet,1
```

## Contact

email: [jonathan.schwartz@czii.org](jonathan.schwartz@czii.org)



