# E-Commerce Customer Lifetime Value Prediction

This project predicts the Customer Lifetime Value (CLV) for retail customers using an end-to-end machine learning pipeline, including data ingestion, transformation, model training, and deployment. Built for practicing production-quality data science workflows.


## Environment setup

### 1 – Create and activate a virtual environment
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

### 2 – Install project dependencies
pip install -r requirements.txt

### 3 – Run the full pipeline
python -m src.pipeline.predict_controller
