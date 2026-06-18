import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# Blueprint for Missing Values analysers
class MissingValuesTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs missing value analysis on the given Dataframe

        Args:
            df (pd.DataFrame): The DataFrame on which the analysis is to be performed
        """
        self.identify(self.df)
        self.visualise(self.df)

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
    
