# --------------------------------------------------------------------
#  src/components/data_transformation.py
# --------------------------------------------------------------------
import os, sys, logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from src.exception_handling import CustomException
from src.logger import logging


# 1. ------------------------------------------------------------------
#    Configuration container
# --------------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join("artifacts", "train_transformed.npy")
    transformed_test_path:  str = os.path.join("artifacts", "test_transformed.npy")


# 2. ------------------------------------------------------------------
#    Transformation component
# --------------------------------------------------------------------
class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    # ---- step A: aggregate raw RFM (no capping) ---------------------
    @staticmethod
    def _aggregate_rfm(df: pd.DataFrame) -> pd.DataFrame:
        df["Quantity"]    = pd.to_numeric(df["Quantity"],   errors="coerce")
        df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"],  errors="coerce")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df.dropna(subset=["CustomerID"])
        df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

        snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        return (
            df.groupby("CustomerID")
              .agg(Monetary=("TotalPrice", "sum"),
                   Frequency=("InvoiceNo", "nunique"),
                   Recency=("InvoiceDate",
                            lambda x: (snapshot - x.max()).days))
              .reset_index()
        )

    # ---- step B: cap & log-transform using *given* thresholds -------
    @staticmethod
    def _cap_and_transform(rfm: pd.DataFrame,
                           cap_m: float,
                           cap_f: float) -> pd.DataFrame:
        rfm["Monetary_cap"]  = rfm["Monetary"].clip(upper=cap_m)
        rfm["Frequency_cap"] = rfm["Frequency"].clip(upper=cap_f)

        rfm["Monetary_log"]  = np.log1p(rfm["Monetary_cap"])
        rfm["Frequency_log"] = np.log1p(rfm["Frequency_cap"])
        rfm["Recency_log"]   = np.log1p(rfm["Recency"])

        return rfm

    # ---- public entry-point ----------------------------------------
    def initiate_data_transformation(
        self, train_csv: str, test_csv: str
    ) -> Tuple[str, str]:
        logging.info("🚀 Data-transformation step started")

        try:
            train_df = pd.read_csv(train_csv)
            test_df  = pd.read_csv(test_csv)

            # 1. build raw RFM on training data
            train_raw = self._aggregate_rfm(train_df)

            # 2. learn caps *only* from training split
            cap_m = train_raw["Monetary"].quantile(0.99)
            cap_f = train_raw["Frequency"].quantile(0.99)

            # 3. apply caps & logs
            train_rfm = self._cap_and_transform(train_raw, cap_m, cap_f)
            test_raw  = self._aggregate_rfm(test_df)
            test_rfm  = self._cap_and_transform(test_raw,  cap_m, cap_f)

            # 4. persist as NumPy arrays
            os.makedirs(os.path.dirname(self.config.transformed_train_path),
                        exist_ok=True)

            np.save(self.config.transformed_train_path, train_rfm.values)
            np.save(self.config.transformed_test_path,  test_rfm.values)

            logging.info("✅ Data-transformation completed")
            return (self.config.transformed_train_path,
                    self.config.transformed_test_path)

        except Exception as e:
            logging.error("❌ Error during data-transformation")
            raise CustomException(e, sys)


# 3. ------------------------------------------------------------------
#    Manual test-run
# --------------------------------------------------------------------
if __name__ == "__main__":
    paths = (os.path.join("artifacts", "train.csv"),
             os.path.join("artifacts", "test.csv"))

    DataTransformation().initiate_data_transformation(*paths)