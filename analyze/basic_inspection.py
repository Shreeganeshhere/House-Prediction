from abc import ABC, abstractmethod
import pandas as pd


# Abstract base class for data inspection strategies
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Abstract method to inspect the data.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: The method should print the inspection results directly.
        """
        pass

# Concrete Strategy for Data Types Inspection
# Strategy that inspects the data types and non-null counts of the dataframe columns
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Inspects and prints data types and non null counts of the dataframe columns.

        Args:
            df (pd.DataFrame): The dataframe to be inspected.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())

# Concrete Strategy for Summary Statistics Inspection
# Stratergy to generate statistical summary of the data(count, mean, std, min....)
class SummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Inspects and prints statistical summary for numerical and categorical features of the dataframe

        Args:
            df (pd.Dataframe): The dataframe to be inspected
        """
        print("\nSummary of the numerical columns")
        print(df.describe())
        print("\nSummary of Categorical columns")
        print(df.describe(include=['object']))


# Context class that uses the Data Inspection strats
# Allows to switch between data inspection stratergies
class DataInspector():
    def __init__(self, strategy: DataInspectionStrategy):
        """Intializes the Data Inspector with the provided strategy

        Args:
            strategy (DataInspectStrategy): The strategy to be used
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataInspectionStrategy):
        """Sets the Data Inspector with a new strategy

        Args:
            strategy (DataInspectionStrategy): The new strategy to use
        """
        self._strategy = strategy
    
    def execute_inspection(self, df: pd.DataFrame):
        """Executes the inspection with respect to the set strategy

        Args:
            df (pd.DataFrame): The dataframe on which the inspection is to be performed
        """
        self._strategy.inspect(df)


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Extracted_data/AmesHousing.csv")
    
    # Set up an inspector
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)

    # change strategy
    inspector.set_strategy(SummaryInspectionStrategy())
    inspector.execute_inspection(df) 