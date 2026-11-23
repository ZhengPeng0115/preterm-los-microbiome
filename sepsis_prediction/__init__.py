"""
Sepsis Prediction Pipeline
==========================
Modular, split-aware machine learning framework for late-onset sepsis (LOS)
and pathogen-specific prediction using clinical and multi-modal microbiome data.
"""

from .config import PipelineConfig
from .preprocessing import ClinicalSplitPreprocessor, filter_low_prevalence_taxa, process_microbiome
from .data_loader import SepsisDataLoader
from .feature_selection import stratified_shap_stability_select
from .modeling import train_xgboost_bayes, evaluate_model
from .evaluation import (
    compute_binary_metrics,
    bootstrap_pooled_metrics,
    select_threshold_max_spec_slope,
    custom_confusion_matrix,
)
from .visualization import (
    plot_probability_distribution,
    plot_performance_curves,
    plot_roc_curve,
    plot_multi_confusion,
    plot_auc_heatmap,
    build_union_and_align_shap,
    plot_shap_beeswarm,
)
from .experiments import (
    run_case_control_study,
    run_full_cohort_study,
)

__all__ = [
    "PipelineConfig",
    "ClinicalSplitPreprocessor",
    "filter_low_prevalence_taxa",
    "process_microbiome",
    "SepsisDataLoader",
    "stratified_shap_stability_select",
    "train_xgboost_bayes",
    "evaluate_model",
    "compute_binary_metrics",
    "bootstrap_pooled_metrics",
    "select_threshold_max_spec_slope",
    "custom_confusion_matrix",
    "plot_probability_distribution",
    "plot_performance_curves",
    "plot_roc_curve",
    "plot_multi_confusion",
    "plot_auc_heatmap",
    "build_union_and_align_shap",
    "plot_shap_beeswarm",
    "run_case_control_study",
    "run_full_cohort_study",
]
