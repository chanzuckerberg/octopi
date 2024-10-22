# cziimaginginstitute-model-exploration
Codebase for CZII's 3d particle picking model exploration project.

## Installation and setup
Inside the directory, run `pip install -e .` 

To use CZI cloud MLflow tracker, add a `.env` in the root directory like below. You can get an MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token) (please note that a new token will be generated everytime you open this site).
```
MLFLOW_TRACKING_USERNAME = Your_CZ_email
MLFLOW_TRACKING_PASSWORD = Your_mlflow_access_token
```

## Training with generic PyTorch  
```
python3 /src/model_explore/train.py 
    --copick_config_path your_copick_config_path \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --num_random_samples_per_batch 16 \
    --learning_rate 1e-4 \
    --num_epochs 200
```

## Training with PyTorch Lighting (distributed training)
```
python3 /src/model_explore/train_pl.py 
    --copick_config_path your_copick_config_path \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --num_random_samples_per_batch 16 \
    --learning_rate 1e-4 \
    --num_gpus 4 \
    --num_epochs 200 
```

## MLflow tracking 
To view the tracking results, go to a deployed [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). Note the project name needs to be registered first.