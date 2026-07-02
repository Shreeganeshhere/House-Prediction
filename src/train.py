import json
import pandas as pd
import joblib, os, math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train(data_dir: str="data/processed", model_dir: str="models"):
    """
    Trains a Linear Regression model and saves it

    Args:
        data_dir (str, optional): The path where the data resides. Defaults to "data/processed".
        model_dir (str, optional): The path to where the model will be saved. Defaults to "models".
    """
    # Make model directory if does not exist
    os.makedirs(model_dir, exist_ok=True)

    # Import the transformed data
    x_train = pd.read_csv(f"{data_dir}/x_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    x_test = pd.read_csv(f"{data_dir}/x_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict on the test set
    preds = model.predict(x_test)

    # Evaluate on Root mean squared error and r2
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

    with open("metrics.json", "w") as f:
        json.dump({"rmse": rmse, "r2": r2}, f, indent=2)

    # Save the model
    with open(f"{model_dir}/linear_regression.pkl", "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    train()

