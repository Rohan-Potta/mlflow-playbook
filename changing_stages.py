import mlflow
import os
from mlflow.tracking import MlflowClient

# Load DagsHub token from environment variables for secure access
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set MLflow tracking environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token

# DagsHub URL and model details
dagshub_url = "https://dagshub.com"
repo_owner = "httpriyang"
repo_name = "CI_MLops"
model_name = "Best Model"  # Specify your registered model name

# Promote the latest model in Staging to Production and archive the current Production model
def promote_model_to_production():
    """Promote the latest model in Staging to Production and archive the current Production model."""
    client = MlflowClient()

    # Get the latest model in the Staging stage
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model found in the 'Staging' stage.")
        return

    latest_staging_version = staging_versions[0]
    staging_version_number = latest_staging_version.version

    # Get the current Production model, if any
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    if production_versions:
        current_production_version = production_versions[0]
        production_version_number = current_production_version.version

        # Transition the current Production model to Archived
        client.transition_model_version_stage(
            name=model_name,
            version=production_version_number,
            stage="Archived",
            archive_existing_versions=False
        )
        print(f"Archived model version {production_version_number} in 'Production'.")
    else:
        print("No model currently in 'Production'.")

    # Transition the latest Staging model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version_number,
        stage="Production",
        archive_existing_versions=False
    )
    print(f"Promoted model version {staging_version_number} to 'Production'.")

if __name__ == "__main__":
    promote_model_to_production()
