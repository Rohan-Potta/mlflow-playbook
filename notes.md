Run IU -> mlflow ui --host 0.0.0.0 --port 5000

Runs: They are execution of code , for example python train.py, this can store metrics, parameters, start and end times and artifacts (output files from the run such as model weights, images, etc).


Experiments:
An experiment groups together runs for a specific task. 


By default, without any particular server/database configuration, MLflow Tracking logs data to the local mlruns directory.(https://mlflow.org/docs/latest/tracking/#tracking-setup) to set up a remote database to store the results 


We can use the Mllfow client to get access of the tracking UI programmatically , for example trying to find the best run present in the experiment 

```python
    client = mlflow.tracking.MlflowClient()
    experiment_id = "0"
    best_run = client.search_runs(
        experiment_id, order_by=["metrics.val_loss ASC"], max_results=1
    )[0]
    print(best_run.info)
    {'run_id': '...', 'metrics': {'val_loss': 0.123}, ...}
```

using

```python
mlflow.log_artifact("artifactname.pkl")
mlflow.log_artifact("picture.png")
mlflow.log_artifact(__file__) #this is used to track the file we execute

```

```python
mlflow.set_tracking_uri("http://<mlflow-server-host>:<port>") #use this commmand if you want to set up to a remote server
```

This is used to get the inputs and outputs to the model and tracks it to mlflow
```python 
from mlflow.models.signature import infer_signature

# Suppose you have training data X and model output y_pred
signature = infer_signature(X, y_pred)

# Then use it when logging the model
mlflow.sklearn.log_model(model, "model", signature=signature)
```


ML Flow Model Registry:

Model
An MLflow Model is created from an experiment or run that is logged with one of the model flavor’s mlflow.<model_flavor>.log_model() methods. 

Registered Model
An MLflow Model can be registered with the Model Registry. A registered model has a unique name, contains versions, aliases, tags, and other metadata.

Model Version
Each registered model can have one or many versions. When a new model is added to the Model Registry, it is added as version 1. Each new model registered to the same model name increments the version number. Model versions have tags, which can be useful for tracking attributes of the model version (e.g. pre_deploy_checks: "PASSED")


