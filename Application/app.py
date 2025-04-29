from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Model and scaler loading from MLflow Model Registry
model_name = "Linear_Regression_Model"
model_version = 1



scaler = joblib.load('scaler.pkl')
# model = joblib.load('linear_regression_model.pkl')
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get user input from form (MedInc, HouseAge, AveRooms)
            features = [
                float(request.form["MedInc"]),
                float(request.form["HouseAge"]),
                float(request.form["AveRooms"])
            ]
            
            # Scale the features using the saved scaler
            features_scaled = scaler.transform([features])
            input_df = pd.DataFrame(features_scaled, columns=['housing_median_age', 'total_rooms', 'median_income'])
            # Make the prediction
            prediction = model.predict(input_df)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
        
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    # Run the app on 0.0.0.0 and port 7777
    app.run(host='0.0.0.0', port=7777, debug=True)
