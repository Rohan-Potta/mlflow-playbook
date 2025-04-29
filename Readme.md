ML - Flow Tutorial https://www.youtube.com/watch?v=GlvgqliaQaA&t=1s

This is used for experiment tracking 

We have various process such as:
    1) Pre-Processing
    2) Feature Engineernig 
    3) Model Training
    4) Parameter Tuning

And within each of these we have different Techniques or methods and we dont know which one is the correct or the best one , so for that we need to note all the methods and the results and for that we need to keep track of the experiments and for that we use ML-Flow

DVC VS ML-Ops:
    Use DVC for data versioning + pipeline stages.
    Use MLflow for tracking metrics, parameters, and registering models.

| **Component**          | **MLflow**                                      | **DVC**                                        |
|------------------------|-------------------------------------------------|------------------------------------------------|
| **Experiment Tracking** | ✅ Parameters, metrics, artifacts, UI            | ⚠️ Not primary; can track manually             |
| **Model Versioning**   | ✅ With built-in model registry                  | ✅ Uses Git + remote storage                   |
| **Data Versioning**    | ⚠️ Minimal (log artifacts only)                  | ✅ Core feature                                |
| **Pipeline Management**| 🟡 Limited (with MLflow Projects)                | ✅ Strong DAG-based pipeline support           |
| **Storage Backend**    | File system, S3, GCS, etc.                       | Any cloud remote + Git                         |
| **Integration with Git**| ❌ Not tightly coupled (Works independent)      | ✅ Strong Git integration                      |

To install:
    pip install mlflow
    mlflow ui => used to give the UI and starts the UI

Experiment: The "Folder" or Project
Think of an experiment as a container or project workspace. It groups multiple related model training attempts (aka runs) together.

You define an experiment once.
Each time you try a different model configuration, it creates a new run within that experiment.


Run: A Single Execution
A run is one single execution of your training pipeline — with specific parameters, code, metrics, and artifacts.




To initialize DVC:
```bash 
dvc init
dvc add path/to/file_or_directory
git add data/train.csv.dvc .gitignore
git commit -m "Track train.csv with DVC"
```

Never git add the large data file itself — only the .dvc file. 

```bash
dvc push
```
This will push it to the remote configured, pushes the data

```bash
dvc remote add -d remote_name path:to:storage 
```

dvc fetch: to get the dataset from the remote   

dvc repro-> Use this to run the dvc pipeline

To vsualize the DVC:
we can use "dvc dag" or "dvc pipeline show"


