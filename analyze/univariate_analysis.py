import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Blueprint for Univariate analyzers
class Univariate(ABC):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univariate analysis on the specified feature of the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analyzed
            feature (str): The feature in the dataframe to be analysed
        """
        pass


# Concrete class of Univariate analyser on numerical features
class NumericalUnivariateAnalyser(Univariate):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univariate analysis on the specified numerical features of the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed
            feature (str): The feature in the dataframe to be analysed
        """
        plt.figure(figsize=(12,8))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

# Concrete class of Univariate analyser on categorical features
class CategoricalUnivatiateAnalyser(Univariate):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univaritate analysis on the specified categorical features of the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed 
            feature (str): The feature in the dataframe to be analysed
        """
        plt.figure(figsize=(12,8))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.xticks(rotation=45)
        plt.show()



class UnivariateAnalyzer:
    def __init__(self, strategy: str):
        """Initializes strategy with the specified strategy

        Args:
            strategy (str): The strategy to use
        """
        self._strategy = strategy
    

    def change_strategy(self, strategy: str):
        """Changes the old strategy with the new specified strategy

        Args:
            strategy (str): The strategy to use
        """
        self._strategy = strategy
    

    def execute(self, df: pd.DataFrame, feature: str):
        """Executes the strategy on the specified feature of the given dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed 
            feature (str): The feature in dataframe to be analysed
        """
        self._strategy.analyze(df, feature)



if __name__ == "__main__":

    df = pd.read_csv('./Extracted_data/AmesHousing.csv')

    # Analyzing a numerical feature
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalyser())
    # analyzer.execute(df, 'SalePrice')

    # Analyzing a categorical feature
    analyzer.change_strategy(CategoricalUnivatiateAnalyser())
    analyzer.execute(df, 'Neighborhood')