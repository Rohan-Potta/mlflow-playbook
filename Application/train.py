import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# Set up the experiment
experiment_name = "linear_regression"
experiment_id = mlflow.set_experiment(experiment_name)

# Start MLflow run
with mlflow.start_run(run_name="linear_regression_v1"):
    # Load the dataset
    df = pd.read_csv('housing.csv')

    # Select 3 features for the model
    X = df[['housing_median_age', 'total_rooms', 'median_income']]  # Update based on dataset
    y = df['median_house_value']  # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    signature = infer_signature(X_test, y_pred)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    mlflow.log_metric("mse", mse)

    # Save the model and scaler to files
    joblib.dump(model, 'linear_regression_model.pkl')
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature, #this is used to track the inputs and the outputs of the model
        registered_model_name="Linear_Regression_Model",
        )

    mlflow.log_artifact(__file__)

    
    # Save and log the scaler artifact
    joblib.dump(scaler, 'scaler.pkl')0
    mlflow.log_artifact('scaler.pkl')
