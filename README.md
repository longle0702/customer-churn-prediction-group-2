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
make xai        # generate SHAP explainability outputs → reports/figures/xai/
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

## Part 3 — Explainable AI with SHAP

Task 3 implements Explainable AI for the trained LightGBM churn model using
[`shap.TreeExplainer`](https://shap.readthedocs.io/). The implementation is
split according to the Jira tickets used by the group:

| Jira ticket | Scope | Main deliverables |
|-------------|-------|-------------------|
| `KAN-9` | SHAP integration and global interpretation | TreeExplainer, Shapley values, global summary bar, beeswarm, mean SHAP plot |
| `KAN-10` | Localized instance interpretability plots | Waterfall plot, force plot HTML, dependence plots for top features |
| `KAN-11` | Unified report assembly and code sharing | README/report guidance, MLproject entry point, Makefile command, tracked XAI output folder |

### Generate SHAP outputs

Run the regular Part 2 pipeline first so that the processed features and model
exist:

```bash
make data
make features
make train
```

Then generate all XAI artefacts:

```bash
make xai
```

Equivalent direct command:

```bash
python -m customer_churn.explainability \
  --mode all \
  --point-index 0 \
  --max-samples 500 \
  --top-n-dependence 3
```

Equivalent MLflow Project command:

```bash
mlflow run . --env-manager=local -e explain \
  -P mode=all \
  -P point_index=0 \
  -P max_samples=500 \
  -P top_n_dependence=3
```

Generated outputs are saved in:

```text
reports/figures/xai/
├── shap_summary_bar_global.png
├── shap_beeswarm_global.png
├── shap_mean_global.png
├── shap_waterfall_point_0.png
├── shap_force_point_0.html
└── shap_dependence_<feature>.png
```

### Suggested report text

The report should explain that SHAP values quantify each feature's contribution
to the LightGBM churn prediction. The global plots identify the features that
drive churn predictions overall, while the local plots explain why a specific
customer was predicted as high or low churn risk. These explanations describe
model behaviour and should not be interpreted as causal business conclusions.

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
