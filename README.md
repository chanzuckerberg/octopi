# cziimaginginstitute-model-exploration
Codebase for CZII's 3d particle picking model exploration project.

## Installation and setup the environment
Inside the directory, run `pip install -e .` 

To use CZI cloud MLflow tracker, add a `.env` in the root directory like below. You can get a CZI MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token) (note that a new token will be generated everytime you open this site).
```
MLFLOW_TRACKING_USERNAME = Your_CZ_email
MLFLOW_TRACKING_PASSWORD = Your_mlflow_access_token
```

## Prepare the dataset 
Generate picks segmentations for dataset 10439 from the CZ cryoET Dataportal (only need to run this step once). 
```
python3 /src/model_explore/segmentation_from_picks.py 
    --copick_config_path copick_config_dataportal_10439.json \
    --copick_user_name user0 \
    --copick_segmentation_name paintedPicks
```

## Training a 3d U-Net model with generic PyTorch  
```
python3 /src/model_explore/train.py 
    --copick_config_path copick_config_dataportal_10439.json \
    --copick_user_name user0 \
    --copick_segmentation_name paintedPicks \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --num_random_samples_per_batch 16 \
    --learning_rate 1e-4 \
    --num_epochs 100
```

## Training a 3d U-Net model with PyTorch Lighting (distributed training)
```
python3 /src/model_explore/train_pl.py 
    --copick_config_path copick_config_dataportal_10439.json \
    --copick_user_name user0 \
    --copick_segmentation_name paintedPicks \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --num_random_samples_per_batch 16 \
    --learning_rate 1e-4 \
    --num_gpus 4 \
    --num_epochs 100 
```

## Model hyperparameter tuning with Optuna and PyTorch Lightning 
```
python3 /src/model_explore/optuna_pl_ddp.py 
    --copick_config_path copick_config_dataportal_10439.json \
    --copick_user_name user0 \
    --copick_segmentation_name paintedPicks \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --num_random_samples_per_batch 16 \
    --learning_rate 1e-4 \
    --num_gpus 4 \
    --num_epochs 100 \
    --num_optuna_trials 50 
```

## MLflow tracking 
To view the tracking results, go to the CZI [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). Note the project name needs to be registered first.