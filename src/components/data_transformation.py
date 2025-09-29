# --------------------------------------------------------------------
#  src/components/data_transformation.py
# --------------------------------------------------------------------
import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from src.exception_handling import CustomException
from src.logger            import logging


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

    # ---- step A: per-customer RFM aggregation ------------------------
    @staticmethod
    def _aggregate_rfm(df: pd.DataFrame) -> pd.DataFrame:
        df["Quantity"]    = pd.to_numeric(df["Quantity"],   errors="coerce")
        df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"],  errors="coerce")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df.dropna(subset=["CustomerID"]).copy()

        df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
        snapshot_date     = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        return (
            df.groupby("CustomerID")
              .agg(Monetary = ("TotalPrice", "sum"),
                   Frequency= ("InvoiceNo",  "nunique"),
                   Recency  = ("InvoiceDate",
                               lambda x: (snapshot_date - x.max()).days))
              .reset_index()
        )

    # ---- step B: cap at 99-th pct & log-transform --------------------
    @staticmethod
    def _cap_and_transform(rfm: pd.DataFrame,
                           cap_m: float,
                           cap_f: float) -> pd.DataFrame:
        rfm["Monetary_cap"]  = rfm["Monetary"].clip(lower=0, upper=cap_m)
        rfm["Frequency_cap"] = rfm["Frequency"].clip(lower=0, upper=cap_f)
        rfm["Recency"]       = rfm["Recency"].clip(lower=0)  # guard negatives

        rfm["Monetary_log"]  = np.log1p(rfm["Monetary_cap"])
        rfm["Frequency_log"] = np.log1p(rfm["Frequency_cap"])
        rfm["Recency_log"]   = np.log1p(rfm["Recency"])

        # keep only clean rows
        return rfm.dropna(subset=["Monetary_log",
                                  "Frequency_log",
                                  "Recency_log"]).reset_index(drop=True)

    # ---- public entry-point -----------------------------------------
    def initiate_data_transformation(
        self, train_csv: str, test_csv: str
    ) -> Tuple[str, str]:
        logging.info("🚀 Data-transformation step started")

        try:
            train_df = pd.read_csv(train_csv)
            test_df  = pd.read_csv(test_csv)

            # 1. raw RFM on training data
            train_raw = self._aggregate_rfm(train_df)

            # 2. learn caps only from train
            cap_m = train_raw["Monetary"].quantile(0.99)
            cap_f = train_raw["Frequency"].quantile(0.99)

            # 3. apply caps + logs to both splits
            train_rfm = self._cap_and_transform(train_raw, cap_m, cap_f)
            test_rfm  = self._cap_and_transform(
                            self._aggregate_rfm(test_df), cap_m, cap_f)

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
