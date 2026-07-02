from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml

def transform(input_path="data/raw/housing.csv", output_dir="data/processed"):
    """
    Transforms the raw data into processed data

    Args:
        input_path (str, optional): The path at which the raw data resides. Defaults to "data/raw/housing.csv".
        output_dir (str, optional): The path to save the processed data. Defaults to "data/processed".
    """
    # make output directory if doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # read csv data at path input_path
    df = pd.read_csv(input_path)

    # Assign all the input features to X and target feature to y
    X = df.drop(columns=['MedHouseVal'])
    y = df["MedHouseVal"]

    # Import parameters from params.yaml
    params = yaml.safe_load(open("params.yaml"))

    # Split the dataset into train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=params['model']['test_size'], random_state = params['model']['random_state']
        )
    
    # Save the split data in separate csvs
    x_train.to_csv(f'{output_dir}/x_train.csv', index=False)
    x_test.to_csv(f'{output_dir}/x_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    print(f"Train size: {x_train.shape}, Test size: {x_test.shape}")


if __name__ == "__main__":
    transform()