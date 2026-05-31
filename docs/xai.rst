Explainable AI with SHAP
========================

Part 3 adds Explainable AI outputs for the trained LightGBM churn model.

Jira ticket mapping
-------------------

* ``KAN-9`` — SHAP integration and global model interpretation.
* ``KAN-10`` — localized instance interpretability plots.
* ``KAN-11`` — unified report assembly and final code sharing.

Commands
--------

Generate all SHAP outputs after running the Part 2 pipeline::

   make data
   make features
   make train
   make xai

Or run the module directly::

   python -m customer_churn.explainability --mode all --point-index 0

Outputs
-------

The generated artefacts are written to ``reports/figures/xai/``:

* ``shap_summary_bar_global.png``
* ``shap_beeswarm_global.png``
* ``shap_mean_global.png``
* ``shap_waterfall_point_0.png``
* ``shap_force_point_0.html``
* ``shap_dependence_<feature>.png``

Interpretation guidance
-----------------------

Use the summary, beeswarm and mean SHAP plots for global interpretation of the
model. Use the waterfall and force plots to explain one selected customer. Use
dependence plots to discuss how the most influential features relate to churn
risk in the model.