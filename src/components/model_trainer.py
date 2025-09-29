# src/components/model_trainer.py
import os, sys, json, pickle
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.logger import logging
from src.exception_handling import CustomException


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
@dataclass
class ModelTrainerConfig:
    model_path:  str = os.path.join("artifacts", "model.pkl")
    metric_path: str = os.path.join("artifacts", "cv_metrics.json")


# ------------------------------------------------------------------ #
#  Trainer with cross-validation
# ------------------------------------------------------------------ #
class ModelTrainer:
    def __init__(self) -> None:
        self.config = ModelTrainerConfig()

    # hyper-params (pulled from Optuna notebook)
    @staticmethod
    def _xgb_params() -> dict:
        return {
            "learning_rate": 0.011871477706268556,
            "max_depth": 3,
            "n_estimators": 455,
            "subsample": 0.7720810091774672,
            "colsample_bytree": 0.8566474552304917,
            "alpha": 0.014812603546710549,
            "lambda": 0.09058337322381116,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
        }

    # main entry
    def initiate_model_trainer(
        self,
        data_arr:  np.ndarray,
        n_splits:  int = 5
    ) -> Tuple[str, str]:
        logging.info("🚀 Cross-validation started")

        try:
            X, y = data_arr[:, :-1], data_arr[:, -1]
            kf   = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            fold_metrics: List[dict] = []

            for fold, (tr, val) in enumerate(kf.split(X), start=1):
                model = xgb.XGBRegressor(**self._xgb_params())
                model.fit(X[tr], y[tr])

                preds = model.predict(X[val])

                metrics = {
                    "fold": fold,
                    "MAE":  round(mean_absolute_error(y[val], preds), 2),
                    "RMSE": round(root_mean_squared_error(y[val], preds, squared=False), 2),
                    "R2":   round(r2_score(y[val], preds), 4),
                }
                logging.info(f"Fold {fold}: {metrics}")
                fold_metrics.append(metrics)

            # save metrics for every fold
            os.makedirs(os.path.dirname(self.config.metric_path), exist_ok=True)
            with open(self.config.metric_path, "w") as f:
                json.dump(fold_metrics, f, indent=2)

            # persist last-fold model for quick demo
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            with open(self.config.model_path, "wb") as f:
                pickle.dump(model, f)

            logging.info("✅ Cross-validation finished")
            return self.config.model_path, self.config.metric_path

        except Exception as e:
            logging.error("❌ Error during cross-validation")
            raise CustomException(e, sys)
