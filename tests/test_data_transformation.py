import pandas as pd
from src.components.data_transformation import DataTransformation

def test_aggregate_rfm_basic():
    df = pd.DataFrame({
        "Quantity":[10,5],
        "UnitPrice":[2,3],
        "InvoiceDate":[pd.Timestamp("2025-01-01"),
                       pd.Timestamp("2025-01-10")],
        "CustomerID":[1,1],
        "InvoiceNo":["001","002"]
    })
    rfm = DataTransformation._aggregate_rfm(df)
    assert "Monetary" in rfm.columns
    assert rfm["Monetary"].iloc[0] == 10*2 + 5*3
