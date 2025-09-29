# src/components/model_trainer.py
import os, sys, json, pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from src.logger import logging
from src.exception_handling import CustomException

@dataclass
class ModelTrainerConfig:
    model_path: str   = os.path.join("artifacts", "model.pkl")
    metric_path: str  = os.path.join("artifacts", "metrics.json")

class ModelTrainer:
    def __init__(self) -> None:
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr:  np.ndarray,
        target_log: bool = True
    ) -> Tuple[str, str]:
        logging.info("🚀 Model-training step started")

        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:,  :-1], test_arr[:,  -1]

            # best params copied from notebook / Optuna
            params = {
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

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # inverse-transform if target was log-scaled
            if target_log:
                preds_orig = np.expm1(preds)
                y_test_orig = np.expm1(y_test)
            else:
                preds_orig = preds
                y_test_orig = y_test

            metrics = {
                "MAE": round(mean_absolute_error(y_test_orig, preds_orig), 2),
                "RMSE": round(np.sqrt(mean_squared_error(y_test_orig, preds_orig)), 2),
                "R2": round(r2_score(y_test_orig, preds_orig), 4),
            }
            logging.info(f"✅ Metrics: {metrics}")

            # save model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            with open(self.config.model_path, "wb") as f:
                pickle.dump(model, f)

            # save metrics
            with open(self.config.metric_path, "w") as f:
                json.dump(metrics, f, indent=2)

            return self.config.model_path, self.config.metric_path

        except Exception as e:
            logging.error("❌ Error during model training")
            raise CustomException(e, sys)
