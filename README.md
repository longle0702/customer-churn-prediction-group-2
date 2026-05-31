# Customer Churn Prediction

E-Commerce customer churn prediction system – **EPITA AI Project Methodology 2025-2026, Group 2**.

Built with **LightGBM** (GPU-accelerated when available) following the
[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.

---

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

---

## Quickstart

### 1 – Install dependencies
```bash
pip install -r requirements.txt
# or, with dev extras:
pip install -e ".[dev]"
```

### 2 – Run the full pipeline
```bash
make all
# or step by step:
make data       # raw Excel → data/interim/ecommerce_interim.csv
make features   # impute, encode, scale → data/processed/features.csv + target.csv
make train      # train LightGBM, print metrics
make predict    # inference → data/processed/predictions.csv
make plots      # generate all figures → reports/figures/
```

### 3 – Run tests
```bash
make test
```

### 4 – Lint & format
```bash
make lint     # flake8
make format   # black
```

---

## Evaluation Metrics

Training and inference both report:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted churners, fraction that truly churn |
| **Recall** | Of true churners, fraction that are caught |
| **F1-score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |
| **Inference time (ms/sample)** | Per-prediction latency |

---

## GPU Support

Device auto-detected in [`customer_churn/config.py`](customer_churn/config.py):

- **Apple Silicon (M-series)** → OpenCL via Metal (`device="gpu"`)
- **NVIDIA GPU** → CUDA (`device="gpu"`)
- **CPU fallback** – applied automatically if GPU build is unavailable

---

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Dataset – Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
