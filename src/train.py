import json
import pandas as pd
import joblib, os, math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import yaml

# def train(data_dir: str="data/processed", model_dir: str="models"):
#     """
#     Trains a Linear Regression model and saves it

#     Args:
#         data_dir (str, optional): The path where the data resides. Defaults to "data/processed".
#         model_dir (str, optional): The path to where the model will be saved. Defaults to "models".
#     """
#     # Make model directory if does not exist
#     os.makedirs(model_dir, exist_ok=True)

#     # Import the transformed data
#     x_train = pd.read_csv(f"{data_dir}/x_train.csv")
#     y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
#     x_test = pd.read_csv(f"{data_dir}/x_test.csv")
#     y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

#     # Train the Linear Regression model
#     model = LinearRegression()
#     model.fit(x_train, y_train)

#     # Predict on the test set
#     preds = model.predict(x_test)

#     # Evaluate on Root mean squared error and r2
#     rmse = math.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)
#     print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

#     with open("metrics.json", "w") as f:
#         json.dump({"rmse": rmse, "r2": r2}, f, indent=2)

#     # Save the model
#     with open(f"{model_dir}/linear_regression.pkl", "wb") as f:
#         joblib.dump(model, f)

def train(data_dir: str="data/processed", model_dir: str="models"):
    """
    Trains a XGBoost model"

    Args:
        data_dir (str, optional): The path where the data resides. Defaults to "data/processed".
        model_dir (str, optional): The path where the model is to be saved. Defaults to "models".
    """
    # Make directory to save model if doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    # Load parameters from params.yaml
    params = yaml.safe_load(open('params.yaml'))['model']

    # Load the training and test data
    x_train = pd.read_csv(f"{data_dir}/x_train.csv")
    x_test = pd.read_csv(f"{data_dir}/x_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    # Define Pipeline of the model (Scaler -> XGBRegressor)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"]
        ))
    ])

    # Set the mlflow experiment name
    mlflow.set_experiment("house-price-predictor")
    
    # Start the mlflow autologging
    # mlflow.sklearn.autolog()

    # Start the mlflow run
    with mlflow.start_run():
        # Fit the pipeline on the training data
        pipeline.fit(x_train, y_train)
        # Predict on the test data
        preds = pipeline.predict(x_test)

        # Metrics for evaluation (Root Mean Squared Error and R2 Score)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"RMSE: {rmse: .4f}\tR2: {r2: .4f}")

        # Log parameters, metrics and model to mlflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipeline, "pipeline",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"])

        # Save metrics to json file
        with open("metrics.json", "w") as f:
            json.dump({"rmse": round(rmse, 4), "r2": round(r2, 4)}, f)

        # Save the pipeline to a pickle file    
        with open(f"{model_dir}/models.pkl", "wb") as f:
            joblib.dump(pipeline, f)

if __name__ == "__main__":
    train()

