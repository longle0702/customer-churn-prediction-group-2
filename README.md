# Customer Churn Prediction

E-Commerce customer churn prediction system – **EPITA AI Project Methodology 2025-2026, Group 2**.

Built with **LightGBM** (GPU-accelerated when available) following the
[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.

---

## Project Structure

```
customer-churn-prediction/
├── LICENSE
├── Makefile                             ← Convenience commands (make data, make train …)
├── README.md
├── data
│   ├── external                         ← Data from third-party sources
│   ├── interim                          ← Intermediate / cleaned data
│   │   └── ecommerce_interim.csv
│   ├── processed                        ← Final ML-ready features & predictions
│   │   ├── features.csv
│   │   ├── target.csv
│   │   └── predictions.csv
│   └── raw
│       └── E Commerce Dataset.xlsx      ← Original, immutable data dump
├── docs                                 ← Sphinx documentation
├── models
│   └── lgbm_churn_model.txt             ← Trained LightGBM model (native format)
├── notebooks                            ← Jupyter notebooks for exploration
├── pyproject.toml                       ← Package metadata + tool config (black, pytest)
├── references                           ← Data dictionaries and explanatory materials
├── reports
│   └── figures                          ← Generated plots (PNG)
├── requirements.txt
├── setup.cfg                            ← flake8 configuration
└── customer_churn                       ← Source package
    ├── __init__.py
    ├── config.py                        ← All variables, paths, hyper-parameters, GPU detection
    ├── dataset.py                       ← Step 1: load raw Excel → data/interim/
    ├── features.py                      ← Step 2: feature engineering → data/processed/
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py                     ← Step 3: LightGBM training + evaluation metrics
    │   └── predict.py                   ← Step 4: inference + timing
    └── plots.py                         ← Visualisations → reports/figures/
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
