import os
import pandas as pd
from sklearn.datasets import fetch_california_housing


def ingest_data(output_path: str = "data/raw/housing.csv"):
    """
    Ingests california housing data from sklearn
    Downloads the dataset as a csv to data/raw/housing.csv by default

    Args:
        output_path (str, optional):The path to store the dataset. Defaults to "data/raw/housing.csv".
    """
    # Make directory output_path. If exists dont raise error
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Fetch dataset as pandas Dataframe
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path} - shape: {df.shape}")

if __name__ == "__main__":
    ingest_data()
