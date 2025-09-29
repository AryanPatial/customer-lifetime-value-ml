# src/pipeline/predict_controller.py
import numpy as np, sys
from src.logger import logging
from src.exception_handling import CustomException
from src.components.data_ingestion      import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer       import ModelTrainer


def run() -> None:
    try:
        logging.info("🚀 Pipeline run started")

        # 1 Ingestion
        train_csv, test_csv = DataIngestion().initiate_data_ingestion()
        logging.info(f"✓ Ingestion complete · train → {train_csv}")

        # 2 Transformation
        train_np, test_np = DataTransformation().initiate_data_transformation(
            train_csv, test_csv
        )
        logging.info(f"✓ Transformation complete · arrays saved")

        # 3 Load arrays
        train_arr = np.load(train_np)
        test_arr  = np.load(test_np)     # kept for future hold-out use
        logging.info("✓ Arrays loaded into memory")

        # 4 Cross-validated training
        model_path, metrics_path = ModelTrainer().initiate_model_trainer(train_arr)
        logging.info(f"✓ Training finished · model → {model_path} · metrics → {metrics_path}")

        logging.info("🎉 Pipeline completed successfully")

    except Exception as e:
        logging.error("❌ Pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run()
