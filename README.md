# cziimaginginstitute-model-exploration
Codebase for CZII's 3d particle picking model exploration project.

To run the model, add a `.env` in the root directory like below. You can get an MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/)
```
MLFLOW_TRACKING_USERNAME = Your_CZ_email
MLFLOW_TRACKING_PASSWORD = Your_mlflow_access_token
```

Then use the command `python3 /src/model_explore/train.py`
To view the tracking results, go to a deployed [mlflow server](https://mlflow.cw.use4-prod.si.czi.technology/). Note the project name needs to be registered first.