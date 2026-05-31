.PHONY: all data features train predict plots xai clean lint format pylint test docs help

# -------------------------------------------------------------------
# Variables
# -------------------------------------------------------------------
# When `conda activate churn_project` is active, CONDA_PREFIX is set
# to the env root (e.g. ~/miniforge3/envs/churn_project).
# We use that to pin the exact Python interpreter, falling back to
# whatever python3 is on PATH if no conda env is active.
PYTHON := $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python3)
PKG    := customer_churn

# -------------------------------------------------------------------
# Default target – full pipeline
# -------------------------------------------------------------------
all: data features train predict plots  ## Run the full ML pipeline

# -------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------
data:      ## Step 1 – Load raw Excel → data/interim/
	$(PYTHON) -m $(PKG).dataset

features:  ## Step 2 – Feature engineering → data/processed/
	$(PYTHON) -m $(PKG).features

train:     ## Step 3 – Train LightGBM model (GPU if available)
	$(PYTHON) -m $(PKG).modeling.train

predict:   ## Step 4 – Run inference → data/processed/predictions.csv
	$(PYTHON) -m $(PKG).modeling.predict

plots:     ## Step 5 – Generate all figures → reports/figures/
	$(PYTHON) -m $(PKG).plots

xai:       ## Step 6 – Generate SHAP XAI outputs → reports/figures/xai/
	$(PYTHON) -m $(PKG).explainability --mode all --point-index 0

# -------------------------------------------------------------------
# Code quality
# -------------------------------------------------------------------
lint:      ## Check PEP 8 compliance with flake8
	flake8 $(PKG)/ tests/

format:    ## Auto-format with black
	black $(PKG)/ tests/

pylint:    ## Run pylint static analysis
	pylint $(PKG)/

test:      ## Run the pytest test suite
	pytest tests/ -v --cov=$(PKG) --cov-report=term-missing

# -------------------------------------------------------------------
# Documentation
# -------------------------------------------------------------------
docs:      ## Build Sphinx HTML documentation
	sphinx-apidoc -o docs/api/ $(PKG)/ --force --module-first -e -q
	sphinx-build -b html docs/ docs/_build/html

# -------------------------------------------------------------------
# Housekeeping
# -------------------------------------------------------------------
clean:     ## Remove all generated data, models and figures
	rm -f data/interim/*.csv data/processed/*.csv models/*.txt reports/figures/*.png
	rm -f reports/figures/xai/*.png reports/figures/xai/*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------
help:      ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
