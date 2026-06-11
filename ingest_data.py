import os
from abc import ABC, abstractmethod
import pandas as pd
import zipfile


# Abstract base class for data ingestors
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        "Abstract method to ingest data from a file"
        pass

# Data ingestor for zip files
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        "Extract a zip file and return the contents as a pandas dataframe"
        # Check if file is zip file
        if not file_path.endswith('.zip'):
            raise ValueError("File is not a zip file")

        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("Extracted_data")
        
        # Get a list of extracted files and find the csv files 
        extracted_files = os.listeddir("Extracted_data")
        csv_files = [f for f in extracted_files if f.endswith('.csv')]

        # Check if no csv files found
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in the zip file")

        # Check if multiple csv files found
        elif len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in the zip file")

        csv_file = os.path.join("Extracted_data", csv_files[0])
        df = pd.read_csv(csv_file)

        return df

       
# Data ingestor factory
class DataIngestorFactory:
    @staticmethod
    def get_ingestor(file_path : str) -> DataIngestor:
        "returns the appropriate data ingestor based on the file extension"\
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.zip':
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for the file extension: {}")

