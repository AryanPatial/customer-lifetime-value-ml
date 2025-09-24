import os                    # work with file-system paths
import sys                   # access the current Python process (used by CustomException)
from src.exception_handling import CustomException  # unified error wrapper
from src.logger import logging                     # project-wide logger
import pandas as pd                                 # read CSV into DataFrame
from sklearn.model_selection import train_test_split # split DF into train/test
from dataclasses import dataclass                   # decorator to build config class

@dataclass                                           # auto-generate __init__, __repr__, etc.
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') #  where train.csv will be saved
    test_data_path : str = os.path.join('artifacts', 'test.csv')  #  where test.csv will be saved
    raw_data_path  : str = os.path.join('artifacts', 'data.csv')  #  where full raw.csv will be saved

class DataIngestion:           # component that performs the ingestion step
    def __init__(self):        # runs when you call DataIngestion()
        self.ingestion_config = DataIngestionConfig()  # attach the config so all methods can use it

    def initiate_data_ingestion(self):  # main method you will call from the pipeline
        logging.info("Data Ingestion started")        # log start-time message
        try:
            # ✅ Dynamically build path to Excel file in project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            file_path = os.path.join(project_root, "Online Retail.xlsx")

            df = pd.read_excel(file_path)
            logging.info("Dataset read as pandas dataframe")

            # 2️⃣ make sure folder 'artifacts/' exists before writing any files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3️⃣ persist the untouched data for lineage/debugging
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # 4️⃣ split the DataFrame sitting in RAM
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5️⃣ write the two splits to disk
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of data is completed")

            # 6️⃣ return only the *paths* so downstream steps can reload on demand
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except Exception as e:                         # any error…
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)              # …is wrapped for consistent reporting


if __name__ == "__main__":                            # if you run this script directly
    obj = DataIngestion()                             # create an instance of DataIngestion
    obj.initiate_data_ingestion()                     # call the method to start the ingestion