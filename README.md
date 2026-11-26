# Multi-Modal Sepsis Prediction Pipeline

A modular, leak-free machine learning framework for predicting **Late-Onset Sepsis (LOS)** and **causative pathogen-specific sepsis** in preterm infants using clinical characteristics and longitudinal gut microbiome profiles.

---

## 1. Project Code Architecture

The codebase is organized into a modular package (`sepsis_prediction`), a top-level CLI execution script (`run_pipeline.py`), and an automated test suite:

```
.
├── sepsis_prediction/               # Core pipeline package
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # Pipeline settings, search spaces, and file paths
│   ├── data_loader.py               # Ingestion and multi-modal patient alignment
│   ├── preprocessing.py             # Split-aware clinical imputation, CLR, and longitudinal aggregation
│   ├── feature_selection.py         # Stratified SHAP stability selection & causative pathogen prioritization
│   ├── modeling.py                  # XGBoost Bayesian optimization with dynamic min_child_weight bounds
│   ├── evaluation.py                # Test threshold optimization and 100-run bootstrap 95% CIs
│   ├── visualization.py             # Crash-safe violin plots, ROC curves, confusion matrices & SHAP beeswarm plots
│   └── experiments.py               # Orchestration of Experiments (a), (b), and (c) with target-aware outer CV
│
├── run_pipeline.py                  # Primary CLI script for running single or multiple experiments
└── tests/
    └── test_pipeline.py             # Comprehensive pytest test suite
```

---

## 2. Input Data

The pipeline integrates clinical metadata with longitudinal microbiome profiling across multiple taxonomic resolutions:

| Modality / File | Source Path | Description |
|---|---|---|
| **Clinical Metadata** | `Metadata.xlsx` | Patient-level characteristics (`Hospital_site`, `GA_total`, `Birthweight`, `Apgar_scores`, `AB_exposure`) and sample timing (`Day_of_sampling`, `Day_before_sepsis`, `Match_group`). |
| **Microbiome Peaks** | `peaks_AA.csv` | Mass-spectrometry MALDI-TOF peak abundances. |
| **Microbiome Species** | `species_AA.csv` | Species-level relative taxonomic abundances. |
| **Microbiome Genus** | `genus_AA.csv` | Genus-level aggregated taxonomic abundances. |
| **Microbiome Phylum** | `phylum_AA.csv` | Phylum-level broad taxonomic composition. |

### Data Processing & Feature Engineering
1. **Multi-Modal Patient Alignment**: Retains the intersection of patients present across all five sources.
2. **Compositional Transformation**: Applies total-sum scaling (closure) and zero replacement (`multi_replace`) followed by Centered Log-Ratio (`clr`) transformation.
3. **Longitudinal Aggregation**: Aggregates longitudinal stool samples for each infant into temporal trajectory features: `mean`, `max`, `last` sample before sepsis, and linear `slope` (protected against zero-variance sampling days).
4. **Microbial Diversity**: Computes longitudinal alpha diversity metrics including Shannon diversity ($H$) and Dominance ($D$).
5. **Leak-Free Preprocessing**: Categorical frequency encoding, continuous median imputation, and one-hot encoding are strictly fitted inside training folds via `ClinicalSplitPreprocessor` and transformed on held-out test splits.

---

## 3. Included Experiments

The framework implements three unified experimental protocols:

```
                       [Multi-Modal Aligned Cohort]
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
 (a) Case-Control (1:3)       (b) Full Cohort Test          (c) Full Cohort Test
     5-Fold CV Search             (Target vs All Controls)      (Target vs Controls + Other Sepsis)
  • Clinical                   • Dynamic Best Source from (a) • Dynamic Best Source from (a)
  • Peaks / Species / Genus    • Target-Aware Outer CV splits • Target-Aware Outer CV splits
  • Phylum / Multimodal Comb   • All Cohort Controls          • Controls + Other Sepsis
  └─► Discovers "Best Source"  └─► Tests generalizability     └─► Tests pathogen specificity
```

### (a) Case-Control Study (1:3 Matched Cohort)
* Evaluates each single data source (`clinical`, `microbiome_peak`, `microbiome_species`, `microbiome_genus`, `microbiome_phylum`) and a multimodal combination (`multimodal_comb`) across 5-fold cross-validation.
* Dynamically identifies the **Best Source** for each pathogen target and for overall sepsis (LOS) based on held-out test AUC.
* Generates pooled out-of-fold **SHAP beeswarm summary plots** for **every evaluated data source** (not only the Best Source) for both the overall sepsis model and each pathogen-specific model.

### (b) Full Cohort Test (All Controls)
* Trains on target pathogen cases vs. matched controls (1:3) and evaluates across all cohort controls (excluding other sepsis cases).
* Evaluates whether the identified Best Source generalizes beyond matched infants to the broad NICU control population.

### (c) Full Cohort Test (All Controls + Other Sepsis)
* Evaluates target pathogen cases against the entire cohort: all controls plus all infants with other sepsis types.
* Assesses the pathogen-specificity of the model against confounding non-target infections.

### Key Methodological Safeguards
* **Target-Aware Outer CV**: Balances rare pathogen cases evenly across test folds, preventing fold starvation and zero-positive evaluation folds.
* **Dynamic `min_child_weight` Upper Bounds**: Caps tree-split regularization to `max(1, min(10, n_pos // 2))` to prevent empty trees (`0:leaf=0`) on rare targets.
* **Causative Pathogen Prioritization**: Prioritizes retention of features representing the target pathogen (e.g., *Serratia*, *Klebsiella*, *E. coli*, *Enterobacter*, *S. aureus*) alongside general dysbiosis and clinical markers.
* **Pooled Bootstrap 95% CIs**: Pools test predictions across all folds and executes 100 stratified bootstrap runs to compute 95% Confidence Intervals for AUC, Sensitivity, Specificity, PPV, NPV, and Balanced Accuracy.

---

## 4. Output Files

All output tables and figures are automatically written to `figures/`:

### A. Metrics CSV Table
* **`figures/pooled_metrics_detailed_table.csv`**: Comprehensive results reporting Data Source, Best Source indicator (`Yes`/`No`), optimal threshold, and 95% CIs (`point [lower - upper]`) for AUC, Sensitivity, Specificity, PPV, NPV, and Balanced Accuracy across:
  - Overall Sepsis (LOS) in Experiment (a) across **all evaluated modalities** (`clinical`, `microbiome_peak`, `microbiome_species`, `microbiome_genus`, `microbiome_phylum`, `multimodal_comb`)
  - Each specific pathogen across **all evaluated modalities** in Experiment (a) and using each pathogen's Best Source in Experiments (b) and (c)

### B. Visualization Figures (Exported as vector `.pdf` and companion `.png`)
* **SHAP Beeswarm Summary Plots**:
  - `exp_a_overall_<source>_shap_beeswarm.pdf` / `.png`: Case-control feature impact and directionality for overall sepsis across **all evaluated modalities** (`clinical`, `microbiome_peak`, `microbiome_species`, `microbiome_genus`, `microbiome_phylum`, `multimodal_comb`), with the Best Source labeled in the plot title.
  - `exp_a_<pathogen>_<source>_shap_beeswarm.pdf` / `.png`: Case-control feature importance for each pathogen across **all evaluated modalities** (not only the Best Source), with the Best Source labeled in the plot title.
  - `exp_b_<pathogen>_shap_beeswarm.pdf` / `.png`: Full cohort (all controls) feature importance using each pathogen's Best Source.
  - `exp_c_<pathogen>_shap_beeswarm.pdf` / `.png`: Full cohort (controls + other sepsis) feature importance using each pathogen's Best Source.
* **Performance Curves**: `exp_{b,c}_<pathogen>_performance_curves.pdf` (Sensitivity, Specificity, PPV, NPV vs threshold).
* **ROC Curves**: `exp_{b,c}_<pathogen>_roc_curve.pdf` (ROC curves with bootstrap 95% CI AUC in legend).
* **Probability Distributions**: `exp_{b,c}_<pathogen>_prob_distribution.pdf` (Violin plots with individual jitter and quartile markers).
* **Confusion Matrices**: `exp_{b,c}_<pathogen>_confusion_matrix.pdf` (Class-annotated heatmaps).

---

## 5. Execution Guide

### Prerequisites & Environment Setup

### 1. Quick Verification Run (Fast Testing)
Executes reduced iterations (3 Bayes iterations, 3 CV folds, 10 bootstrap runs) to quickly verify pipeline integrity:
```bash
python run_pipeline.py --quick --experiments a b c
```

### 2. Full Publication Run (All Experiments & 100 Bootstrap Runs)
Executes complete 5-fold outer cross-validation, full Bayesian hyperparameter search, and 100-run bootstrap 95% CIs:
```bash
python run_pipeline.py --experiments a b c
```

### 3. Running Specific Pathogens
Filter execution to specific pathogens of interest:
```bash
python run_pipeline.py --experiments a b c --pathogens "Serratia marcecens" "Escherichia coli" "Klebsiella spp. "
```
*Available default pathogens*: `"Serratia marcecens"`, `"Escherichia coli"`, `"Klebsiella spp. "`, `"Enterobacter cloacae"`, `"Staphylococcus aureus"`.

### 4. Optional CLI Flags
* `--no-shap-beeswarm`: Skips SHAP beeswarm figure generation to minimize run time.
* `--full`: Overrides debugger auto-detection to force full 100-run bootstrap while attached to a debugger.

### 5. Running Unit Tests
To verify all pipeline components (preprocessing, stability selection, dynamic bounds, beeswarm generation):
```bash
pytest tests/test_pipeline.py -v
```
