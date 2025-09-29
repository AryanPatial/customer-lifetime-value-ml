# src/predict_controller.py
import numpy as np
import os, sys
from src.logger import logging
from src.exception_handling import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



def run() -> None:
    try:
        logging.info("🚀 Pipeline run initiated")

        # 1️⃣ Data Ingestion ------------------------------------------------
        logging.info("Step 1 – Data ingestion started")
        train_csv, test_csv = DataIngestion().initiate_data_ingestion()
        logging.info(
            f"Step 1 ✓ Completed | train_csv: {train_csv} | test_csv: {test_csv}"
        )

        # 2️⃣ Data Transformation -----------------------------------------
        logging.info("Step 2 – Data transformation started")
        train_np_path, test_np_path = DataTransformation().initiate_data_transformation(
            train_csv, test_csv
        )
        logging.info(
            f"Step 2 ✓ Completed | train_npy: {train_np_path} | test_npy: {test_np_path}"
        )

        # 3️⃣ Load NumPy arrays -------------------------------------------
        logging.info("Step 3 – Loading transformed arrays from disk")
        train_arr = np.load(train_np_path)
        test_arr = np.load(test_np_path)
        logging.info("Step 3 ✓ Completed | arrays loaded")

        # 4️⃣ Model Training ----------------------------------------------
        logging.info("Step 4 – Model training started")
        model_path, metric_path = ModelTrainer().initiate_model_trainer(
            train_arr, test_arr
        )
        logging.info(
            f"Step 4 ✓ Completed | model: {model_path} | metrics: {metric_path}"
        )

        logging.info("🎉 Pipeline run finished successfully")
        
    except Exception as e:
        logging.error("❌ Pipeline run failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run()
