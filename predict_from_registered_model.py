# This script loads a registered model from MLflow and makes predictions on a sample DataFrame.
import mlflow.pyfunc
import pandas as pd
import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")


data = pd.DataFrame({
   'ph':3.71608,
   'Hardness':204.89045,
   'Solids':20791.318981,
    'Chloramines':7.300212,
    'Sulfate':368.516441,
    'Conductivity':564.308654,
    'Organic_carbon':10.379783,
    'Trihalomethanes':86.99097,
    'Turbidity':2.963135,

},index=[0])


logged_model = 'runs:/8027495603bf4c3b929004b810d7a20f/Best Model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.

print(loaded_model.predict(pd.DataFrame(data)))