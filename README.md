# cziimaginginstitute-model-exploration
Codebase for CZII's 3d particle picking model exploration project.

## Run codes via ssh
### Installation and setup the environment
Inside the directory, run `pip install -e .` 

To use CZI cloud MLflow tracker, add a `.env` in the root directory like below. You can get a CZI MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token) (note that a new token will be generated everytime you open this site).
```
MLFLOW_TRACKING_USERNAME = <Your_CZ_email>
MLFLOW_TRACKING_PASSWORD = <Your_mlflow_access_token>
```

### Prepare the dataset 
Generate picks segmentations for dataset 10439 from the CZ cryoET Dataportal (only need to run this step once). 
```
python3 /src/model_explore/segmentation_from_picks.py 
    --copick_config_path copick_config_dataportal_10439.json \
    --copick_user_name user0 \
    --copick_segmentation_name paintedPicks
```

### Training a 3d U-Net model with generic PyTorch  
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

### Training a 3d U-Net model with PyTorch Lighting (distributed training)
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

### Model hyperparameter tuning with Optuna and PyTorch Lightning 
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

## Submit a container on Coreweave  
```
runai submit --name <job_name> -i ghcr.io/chanzuckerberg/cziimaginginstitute-model-exploration:<tag> 
             --command "python3 src/model_explore/optuna_pl_ddp.py --num_gpus 4" 
             -e MLFLOW_TRACKING_USERNAME=<Your_CZ_email> -e MLFLOW_TRACKING_PASSWORD=<Your_mlflow_access_token> -g 4 
             --preemptible --interactive --existing-pvc claimname=autonomous-3d-particle-picking-pvc,path=/usr/app/data 
```
 
## MLflow tracking   
To view the tracking results, go to the CZI [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). Note the project name needs to be registered first.


## Example results   
We optimized 3d U-Net architecture with 8 tomograms from dataset 10439, 25 Optuna trials, and 100 epoch for each trial.  
```
[I 2024-10-23 20:59:47,927] Trial 69 finished with value: 0.10366249829530716 and parameters: {'num_layers': 4, 'base_channel': 64, 'num_downsampling_layers': 2, 'num_res_units': 3}. Best is trial 65 with value: 0.11890766769647598.   
Best trial: 0.11890766769647598   
Best hyperparameters: {'num_layers': 4, 'base_channel': 64, 'num_downsampling_layers': 2, 'num_res_units': 3}   
```  
This gives the model structure: `channels [64, 128, 256, 512], strides [2, 2, 1], num_res_units 3`.


