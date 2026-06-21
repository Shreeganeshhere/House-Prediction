import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import seaborn as sns


# Blueprint for Missing Values analysers
class MissingValuesTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs missing value analysis on the given Dataframe

        Args:
            df (pd.DataFrame): The DataFrame on which the analysis is to be performed
        """
        self.identify(df)
        self.visualise(df)

    @abstractmethod
    def identify(self, df: pd.DataFrame):
        """
        Identifies the missing values in the dataset

        Args:
            df (pd.DatFrame): The dataframe for analysis
        """
        pass

    @abstractmethod
    def visualise(self, df: pd.DataFrame):
        """Visualizes the missing values in the dataset

        Args:
            df (pd.DataFrame): the dataframe for analysis
        """
        pass
    
# Concrete class
class SimpleMissingValues(MissingValuesTemplate):
    def identify(self, df: pd.DataFrame):
        """Prints the no. of missing value columnwise

        Args:
            df (pd.DataFrame): The dataframe to be analyzed
        """
        print("Missing Values:\n")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0]) # prints only the columns that have missing values

    def visualise(self, df: pd.DataFrame):
        """Visualizes missing values in a heatmap

        Args:
            df (pd.DataFrame): The dataframe to be analyzed
        """
        print("Visualizing Missing Values")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
        plt.title("MISSING VALUES HEATMAP")
        plt.show()


if __name__ == "__main__":
    # import os
    # print(os.getcwd())
    df = pd.read_csv('./Extracted_data/AmesHousing.csv')
    # Perform Missing Values Analysis
    missing_values_analyzer = SimpleMissingValues()
    missing_values_analyzer.analyze(df)