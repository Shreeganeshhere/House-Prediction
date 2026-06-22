from re import L
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# Blueprint on Bivariate analysers
class Bivariate(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Performs bivariate analysis on feature 1 and 2 of the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed
            feature1 (str): The first feature for analysis
            feature2 (str): The second feature for the analysis
        """
        pass


# Concrete class of Bivariate to analyse two Numerical features
class NumericalBivariateAnalysers(Bivariate):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Performs bivariate analysis on two numerical features.
        Plots on a scatterplot

        Args:
            df (pd.DataFrame): The dataframe to be analysed
            feature1 (str): The first numerical feature for analysis
            feature2 (str): The second numerical feature for analysis
        """
        plt.figure(figsize=(12,8))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} v/s {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete class of Bivariate to analyse two categorical feature
class CategoricalBivariateAnalysers(Bivariate):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perfroms Bivariate analysis on two categorical featues of the dataframe
        Plots on a box graph
        Args:
            df (pd.DataFrame): The dataframe to be analysed
            feature1 (str): The first categorical feature from df
            feature2 (str): The second categorical feature from df
        """
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df, x=feature1, y=feature2)
        plt.title(f"{feature1} v/s {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Class that allows you to change strategy
class BivariateAnalyser():
    def __init__(self, strategy: Bivariate):
        """
        Initializes to the specified strategy

        Args:
            strategy (Bivariate): The strategy to use for analysis
        """
        self._strategy = strategy
    
    def change_strategy(self, strategy: Bivariate):
        """
        Changes strategy to the new specified strategy

        Args:
            strategy (Bivariate): The new strategy to use
        """
        self._strategy = strategy

    def execute(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the analysis with respect to the set strategy

        Args:
            df (pd.DataFrame): The dataframe to be analysed
            feature1 (str): The first feature of df to be analysed 
            feature2 (str): The second feature of df to be analysed
        """
        self._strategy.analyse(df=df, feature1=feature1, feature2=feature2)


# Example usage
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('./Extracted_data/AmesHousing.csv')

    # Analyzing relationship between two numerical features
    analyzer = BivariateAnalyser(NumericalBivariateAnalysers())
    # analyzer.execute(df, 'Gr Liv Area', 'SalePrice')

    # Analyzing relationship between a categorical and a numerical feature
    analyzer.change_strategy(CategoricalBivariateAnalysers())
    analyzer.execute(df, 'Overall Qual', 'SalePrice')
    
