import numpy as np
from src.components.model_trainer import ModelTrainer

def test_trainer_runs():
    arr = np.array([
        [1.0,2.0,3.0,0.5],
        [2.0,1.0,1.5,0.3],
        [0.5,2.5,2.0,0.4]
    ])                           # 3 rows • 3 features + 1 target
    model, metrics = ModelTrainer().initiate_model_trainer(arr, n_splits=2)
    assert model.endswith(".pkl")
    assert metrics.endswith(".json")
