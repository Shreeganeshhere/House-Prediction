from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Blueprint for all the multivariate analysers
class Multivariate(ABC):
    def analyse(self, df: pd.DataFrame):
        """
        Performs multivariate analysis on the dataframe
        generates a correlation heatmap and a pair plot

        Args:
            df (pd.DataFrame): The dataframe to be analysed
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays heatmap of correlations between the features of the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed
        """
        pass
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot of the selected features

        Args:
            df (pd.DataFrame): The dataframe to be analysed
        """
        pass


# Concrete class of Multivariate analyser
class SimpleMultivariate(Multivariate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays correlation heatmap for the numerical features in the dataframe

        Args:
            df (pd.DataFrame): The dataframe to be analysed
        """
        plt.figure(figsize=(12,10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to be analysed
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('./Extracted_data/AmesHousing.csv')

    # Perform Multivariate Analysis
    multivariate_analyzer = SimpleMultivariate()

    # Select important features for pair plot
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    multivariate_analyzer.analyse(selected_features)

