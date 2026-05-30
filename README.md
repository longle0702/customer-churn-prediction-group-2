# customer-churn-prediction-group-2

This is the repository for Group 2 (AIS) in the AI Project Methodology course

## MLOps with MLflow & Serving

This section covers the tracking, reproducibility, and local serving of the Customer Churn Prediction model using MLflow.

### Prerequisites

Ensure you have the required dependencies installed in your environment:

```bash
pip install -r requirements.txt
```

### Training & Tracking with MLproject

To run the training pipeline with reproducible parameter tracking, use the `MLproject` entry point. This bypasses virtual environment isolation to use your active local environment:

```bash
mlflow run . --env-manager=local --experiment-name="Customer_Churn_Experiment" -P n_estimators=120 -P max_depth=6
```

To view the tracked parameters, metrics, and registered models, launch the MLflow UI:

```bash
mlflow ui
```

Then navigate to http://localhost:5000.

### Local Model Serving

To spin up a local REST API endpoint serving the trained model artifact directly, run the following command in a separate terminal:

```bash
mlflow models serve -m "mlruns/1/models/m-2ce30e03b0b84f838819532a6bb5a7a2/artifacts" -p 1234 --env-manager=local
```

### Testing Inference

You can test the running inference server by sending a POST request containing sample features using `curl`:

```bash
curl -X POST http://localhost:1234/invocations \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_split": {
         "columns": ["col1", "col2"],
         "data": [[2.5, 1.8]]
       }
     }'
```
