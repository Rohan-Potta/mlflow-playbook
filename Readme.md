
# **MLflow & DVC: Experiment Tracking and Data Versioning**

## **MLflow Overview**

MLflow is an open-source platform used for managing the end-to-end machine learning lifecycle. It supports:

- Experiment Tracking  
- Project Packaging  
- Model Management  
- Model Deployment  

To install MLflow:

```bash
pip install mlflow
```

To start the MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

This launches a tracking UI where you can view runs, experiments, metrics, parameters, and artifacts.

---

## **MLflow Core Concepts**

### **Experiments**
- Think of an experiment as a *project folder*.
- Groups together multiple *runs* for a specific task.
- Defined once; each run (model attempt) is logged under it.

### **Runs**
- A run is a single execution of training or evaluation code (e.g., `python train.py`).
- Tracks:
  - Parameters
  - Metrics
  - Start and end time
  - Artifacts (model weights, images, logs, etc.)

By default, MLflow logs to a local `mlruns` directory. To use a remote tracking server:

```python
mlflow.set_tracking_uri("http://<mlflow-server-host>:<port>")
```

---

## **Logging Data to MLflow**

### **Logging Artifacts**

```python
mlflow.log_artifact("artifactname.pkl")
mlflow.log_artifact("picture.png")
mlflow.log_artifact(__file__)  # Logs the current script
```

### **Model Input/Output Tracking**

```python
from mlflow.models.signature import infer_signature

# Suppose you have input X and output y_pred
signature = infer_signature(X, y_pred)

mlflow.sklearn.log_model(model, "model", signature=signature)
```

---

## **Programmatic Access with MLflow Client**

You can programmatically access runs, experiments, and metrics:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_id = "0"
best_run = client.search_runs(
    experiment_id, order_by=["metrics.val_loss ASC"], max_results=1
)[0]
print(best_run.info)
```

---

## **MLflow Model Registry**

### **Model**
- Created from a run using `mlflow.<flavor>.log_model()`.

### **Registered Model**
- A named collection of MLflow models, often representing a logical entity (e.g., "churn_model").

### **Model Version**
- Each time a model is registered under a name, it increments the version number.
- Model versions can have:
  - Aliases (e.g., "Staging", "Production")
  - Tags (e.g., `pre_deploy_checks: "PASSED"`)

---

## **When to Use MLflow**

Use MLflow for tracking different stages of an ML pipeline:

1. Pre-Processing  
2. Feature Engineering  
3. Model Training  
4. Hyperparameter Tuning  

Track different techniques and outcomes to determine the best approach.

---

## **DVC Overview**

DVC (Data Version Control) is used to manage and version data, models, and pipelines.

### **Initialize DVC in a Project**

```bash
dvc init
```

### **Track Data Files**

```bash
dvc add path/to/file_or_directory
git add path/to/file.dvc .gitignore
git commit -m "Track file with DVC"
```

**Note**: Never commit the actual data file to Git—only commit the `.dvc` file.

### **Push Data to Remote**

```bash
dvc remote add -d remote_name path/to/remote/storage
dvc push
```

### **Fetching Data**

```bash
dvc fetch
```

### **Running Pipelines**

```bash
dvc repro
```

### **Visualizing Pipelines**

```bash
dvc dag
# or
dvc pipeline show
```

---

## **DVC vs. MLflow**

| **Component**            | **MLflow**                                   | **DVC**                                      |
|--------------------------|----------------------------------------------|----------------------------------------------|
| Experiment Tracking      | ✅ Parameters, metrics, artifacts, UI         | ⚠️ Not primary; manual tracking needed        |
| Model Versioning         | ✅ Built-in model registry                    | ✅ Git-based with remote storage              |
| Data Versioning          | ⚠️ Minimal (log artifacts only)              | ✅ Core feature                               |
| Pipeline Management      | 🟡 Limited (via MLflow Projects)              | ✅ Strong DAG-based pipeline support          |
| Storage Backend          | File system, S3, GCS, etc.                   | Any remote + Git                             |
| Git Integration          | ❌ Not tightly coupled                       | ✅ Strong Git integration                     |

---

## **Useful Tutorial**
- [MLflow Tracking & Registry - YouTube Tutorial](https://www.youtube.com/watch?v=GlvgqliaQaA&t=1s)
