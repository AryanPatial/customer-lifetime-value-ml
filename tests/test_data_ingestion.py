import os
from src.components.data_ingestion import DataIngestion

def test_data_ingestion_creates_files():
    train, test = DataIngestion().initiate_data_ingestion()
    assert os.path.exists(train)
    assert os.path.exists(test)
