Customer Churn Prediction – Documentation
==========================================

E-Commerce customer churn prediction system built with **LightGBM**,
following the Cookiecutter Data Science project structure.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/modules
   xai

Getting Started
---------------

Install the conda environment::

   conda env create -f environment.yml
   conda activate churn_project

Run the full pipeline::

   make all

Generate Part 3 SHAP explainability outputs::

   make xai

The SHAP implementation is available in ``customer_churn.explainability`` and
writes global and local explanation artefacts to ``reports/figures/xai/``.

Run the test suite::

   make test

Lint & format::

   make lint
   make format

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
