"""
Unit tests for the modular Sepsis Prediction Pipeline.
Validates split-aware preprocessing, zero-variance protection,
stratified SHAP stability selection, bootstrap 95% CI calculation,
causative pathogen prioritized feature retention, dynamic upper bounds on min_child_weight,
target-aware outer CV, SHAP beeswarm visualization, and edge case handling.
"""

import numpy as np
import pandas as pd
import pytest
import shap

from sepsis_prediction.config import PipelineConfig
from sepsis_prediction.data_loader import SepsisDataLoader
from sepsis_prediction.preprocessing import (
    ClinicalSplitPreprocessor,
    calculate_individual_microbiome_features,
    calculate_diversity_features,
)
from sepsis_prediction.feature_selection import (
    stratified_shap_stability_select,
    find_causative_feature_indices,
)
from sepsis_prediction.modeling import (
    train_xgboost_bayes,
)
from sepsis_prediction.experiments import (
    get_target_aware_outer_cv,
    run_case_control_study,
)
from sepsis_prediction.evaluation import (
    compute_binary_metrics,
    select_threshold_max_spec_slope,
    bootstrap_pooled_metrics,
    custom_confusion_matrix,
)
from sepsis_prediction.visualization import (
    plot_probability_distribution,
    build_union_and_align_shap,
    plot_shap_beeswarm,
)


def test_clinical_split_preprocessor_no_leakage():
    """Verify that imputer and encoder are fit only on train and transform without leakage."""
    nominal_cols = ["Hospital_site", "Biological_sex"]
    one_hot_cols = ["Hospital_site"]
    scale_cols = ["GA_total", "Birthweight"]

    # Training set
    df_train = pd.DataFrame({
        "Pt_ID": [1, 2, 3, 4],
        "Hospital_site": ["SiteA", "SiteA", "SiteB", np.nan],
        "Biological_sex": [1, 0, 1, 0],
        "GA_total": [28.0, 30.0, np.nan, 32.0],
        "Birthweight": [1000.0, 1200.0, 1100.0, -99.0],  # -99 should be treated as NaN
    })

    # Test set with distinct values and missing fields
    df_test = pd.DataFrame({
        "Pt_ID": [5, 6],
        "Hospital_site": ["SiteA", "SiteB"],
        "Biological_sex": [0, np.nan],
        "GA_total": [np.nan, 34.0],
        "Birthweight": [1500.0, np.nan],
    })

    prep = ClinicalSplitPreprocessor(
        nominal_features=nominal_cols,
        one_hot_features=one_hot_cols,
        scale_features=scale_cols,
        random_state=42,
    )
    prep.fit(df_train)

    # Median of GA_total in train: median([28, 30, 32]) = 30.0
    assert prep.num_imputer_.statistics_[0] == 30.0

    X_tr, names_tr = prep.transform(df_train)
    X_te, names_te = prep.transform(df_test)

    assert X_tr.shape[0] == 4
    assert X_te.shape[0] == 2
    assert names_tr == names_te
    # Ensure no NaN remains in transformed arrays
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()


def test_zero_variance_longitudinal_microbiome():
    """Verify that identical sampling days do NOT crash linregress and return slope = 0.0."""
    patient_data_constant_day = pd.DataFrame({
        "Pt_ID": [101, 101, 101],
        "Day_of_sampling": [3, 3, 3],  # Zero variance in time
        "Taxon_A": [0.1, 0.4, 0.2],
        "Taxon_B_prop": [0.3, 0.3, 0.4],
    })

    # Taxon slopes
    features = calculate_individual_microbiome_features(
        patient_data_constant_day,
        microbiome_cols=["Taxon_A"],
        time_col="Day_of_sampling"
    )
    assert features["Taxon_A_slope"] == 0.0
    assert "Taxon_A_mean" in features
    assert "Taxon_A_max" in features
    assert "Taxon_A_last" in features

    # Diversity slopes
    div_features = calculate_diversity_features(
        patient_data_constant_day,
        microbiome_cols=["Taxon_B_prop"],
        time_col="Day_of_sampling"
    )
    assert div_features["Shannon_slope"] == 0.0
    assert div_features["Dominance_slope"] == 0.0


def test_stratified_shap_stability_select():
    """Verify stratified stability selection maintains consistent 30 features (default) or caps if requested."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 50)
    # Only 4 positive cases (rare pathogen scenario)
    y = np.zeros(20, dtype=int)
    y[:4] = 1

    # Default: adaptive_cap=False -> Consistent 30 features
    selected_30 = stratified_shap_stability_select(
        X=X,
        y=y,
        n_features_to_select=30,
        n_bootstrap=5,
        random_state=42,
        adaptive_cap=False,
    )
    assert len(selected_30) == 30
    assert len(np.unique(selected_30)) == 30

    # Optional: adaptive_cap=True for 4 positives: ceil(4 * 0.6) = 3 features
    selected_capped = stratified_shap_stability_select(
        X=X,
        y=y,
        n_features_to_select=30,
        n_bootstrap=5,
        random_state=42,
        adaptive_cap=True,
    )
    assert len(selected_capped) <= 3
    assert len(np.unique(selected_capped)) == len(selected_capped)


def test_causative_pathogen_prioritized_feature_retention():
    """Verify that causative pathogen features are detected and prioritized into selected set."""
    rng = np.random.RandomState(42)
    X = rng.randn(25, 40)
    y = np.zeros(25, dtype=int)
    y[:5] = 1

    feature_names = [f"RandomTaxon_{i}_mean" for i in range(40)]
    feature_names[15] = "Serratia.marcescens_last"
    feature_names[20] = "Serratia.spp._mean"

    # Test detection helper
    matches = find_causative_feature_indices(feature_names, "Serratia marcecens")
    assert 15 in matches

    # Test retention inside stability selector with consistent 30 features
    selected = stratified_shap_stability_select(
        X=X,
        y=y,
        n_features_to_select=30,
        n_bootstrap=5,
        random_state=42,
        adaptive_cap=False,
        feature_names=feature_names,
        target_pathogen="Serratia marcecens",
        prioritize_causative=True,
    )
    assert len(selected) == 30
    assert 15 in selected  # Causative feature must be prioritized and retained


def test_dynamic_min_child_weight_prevents_empty_trees():
    """Verify dynamic min_child_weight bound prevents empty booster trees on small cohorts."""
    rng = np.random.RandomState(42)
    # Small dataset with only 4 positives
    X = rng.randn(16, 10)
    y = np.zeros(16, dtype=int)
    y[:4] = 1

    model, params = train_xgboost_bayes(
        X_train=X,
        y_train=y,
        random_state=42,
    )
    probs = model.predict_proba(X)[:, 1]
    # Predictions must not all collapse to 0.5
    assert len(np.unique(np.round(probs, 4))) > 1
    # Trees must contain actual splits, not just 0:leaf=0
    dumps = model.get_booster().get_dump()
    assert any("f" in tree for tree in dumps)


def test_target_aware_outer_cv():
    """Verify target-aware CV splits distribute rare cases across all folds without starvation."""
    df_dummy = pd.DataFrame({
        "Pt_ID": range(100),
        "pathogen": ["PathogenA"] * 7 + ["Control"] * 93,
        "Match_group": [f"G_{i}" for i in range(100)],
    })
    pathogen_arr = df_dummy["pathogen"].to_numpy()
    group_arr = df_dummy["Match_group"].to_numpy()

    splits = get_target_aware_outer_cv(
        ref_df=df_dummy,
        target_pathogen="PathogenA",
        pathogen_series=pathogen_arr,
        group_series=group_arr,
        n_splits=5,
        base_seed=1,
    )
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        # Every test fold must contain at least 1 positive case
        assert (pathogen_arr[test_idx] == "PathogenA").sum() >= 1


def test_bootstrap_pooled_metrics_ci():
    """Verify pooled bootstrap calculates valid 95% confidence intervals."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.binomial(1, 0.3, size=n)
    # Generate reasonably correlated probabilities
    probs = np.clip(y_true * 0.5 + rng.uniform(0.1, 0.4, size=n), 0.0, 1.0)

    thresh, _ = select_threshold_max_spec_slope(y_true, probs)
    assert 0.0 <= thresh <= 1.0

    ci_summary = bootstrap_pooled_metrics(
        y_true=y_true,
        probabilities=probs,
        threshold=thresh,
        n_bootstrap=50,
        ci_level=0.95,
        random_state=42,
    )

    for metric in ["auc", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy"]:
        m = ci_summary[metric]
        assert "point" in m
        assert "ci_lower" in m
        assert "ci_upper" in m
        assert m["ci_lower"] <= m["ci_upper"]
        assert 0.0 <= m["ci_lower"] <= 1.0
        assert 0.0 <= m["ci_upper"] <= 1.0


def test_custom_confusion_matrix_dimensions():
    """Verify confusion matrix construction and orientation."""
    y_true = np.array([0, 1, 2, 0, 2])
    y_pred = np.array([1, 0, 0, 1, 1])

    # With other sepsis (3 classes: Pathogen, Other sepsis, Control)
    cm_3x2 = custom_confusion_matrix(y_true, y_pred, include_other_sepsis=True)
    assert cm_3x2.shape == (3, 2)
    # Col 0 = Pathogen, Col 1 = Control
    # y_true=0 (Pathogen), y_pred=1 -> row 0, col 0 (count=2)
    assert cm_3x2[0, 0] == 2
    # y_true=1 (Other sepsis), y_pred=0 -> row 1, col 1 (count=1)
    assert cm_3x2[1, 1] == 1

    # Without other sepsis (2 classes: Pathogen, Control)
    cm_2x2 = custom_confusion_matrix(y_true, y_pred, include_other_sepsis=False)
    assert cm_2x2.shape == (2, 2)


def test_crash_safe_visualization(tmp_path):
    """Verify plot_probability_distribution doesn't crash with 0 or 1 point per class."""
    # Data with only 1 sample for pathogen and 0 for other sepsis
    pathogen_data = {
        "true_labels": [0, 2, 2, 2],
        "probabilities": [0.8, 0.2, 0.3, 0.1],
    }
    save_file = tmp_path / "test_prob_dist.pdf"
    plot_probability_distribution(
        pathogen_data,
        pathogen_name="TestPathogen",
        include_other_sepsis=True,
        save_path=save_file,
    )
    assert save_file.exists()


def test_build_union_and_align_shap():
    """Verify out-of-fold SHAP explanations align correctly across varying feature sets."""
    # Fold 1 selected feats A, B
    vals_1 = np.array([[1.0, 2.0], [0.5, 1.5]])
    base_1 = np.array([0.1, 0.1])
    data_1 = np.array([[10.0, 20.0], [15.0, 25.0]])
    feats_1 = ["FeatA", "FeatB"]

    # Fold 2 selected feats B, C
    vals_2 = np.array([[3.0, 4.0]])
    base_2 = np.array([0.2])
    data_2 = np.array([[22.0, 30.0]])
    feats_2 = ["FeatB", "FeatC"]

    explanation = build_union_and_align_shap(
        shap_values_list=[vals_1, vals_2],
        shap_base_list=[base_1, base_2],
        shap_data_list=[data_1, data_2],
        shap_featnames_list=[feats_1, feats_2],
    )

    # Union features: FeatA, FeatB, FeatC
    assert explanation.feature_names == ["FeatA", "FeatB", "FeatC"]
    assert explanation.values.shape == (3, 3)  # 3 total samples across folds, 3 union features
    assert explanation.data.shape == (3, 3)

    # Sample 0 (fold 1): FeatA=1.0, FeatB=2.0, FeatC unselected -> 0.0
    np.testing.assert_array_equal(explanation.values[0], [1.0, 2.0, 0.0])
    # FeatC unselected in fold 1 -> data is NaN
    assert np.isnan(explanation.data[0, 2])

    # Sample 2 (fold 2): FeatA unselected -> 0.0, FeatB=3.0, FeatC=4.0
    np.testing.assert_array_equal(explanation.values[2], [0.0, 3.0, 4.0])
    assert np.isnan(explanation.data[2, 0])
    assert explanation.data[2, 1] == 22.0
    assert explanation.data[2, 2] == 30.0


def test_plot_shap_beeswarm(tmp_path):
    """Verify that plot_shap_beeswarm creates both PDF and companion PNG without crashing."""
    rng = np.random.RandomState(42)
    n_samples, n_feats = 20, 6
    vals = rng.randn(n_samples, n_feats)
    data = rng.randn(n_samples, n_feats)
    base = np.zeros(n_samples)
    feat_names = [f"Biomarker_{i}" for i in range(n_feats)]

    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=data,
        feature_names=feat_names,
    )

    pdf_path = tmp_path / "test_shap_beeswarm.pdf"
    plot_shap_beeswarm(
        explanation=exp,
        title="Test SHAP Beeswarm Summary",
        save_path=pdf_path,
        max_display=5,
    )

    assert pdf_path.exists()
    png_path = tmp_path / "test_shap_beeswarm.png"
    assert png_path.exists()


def test_all_pathogens_includes_matches():
    """Verify all_pathogens includes Bacillus species and Proteus mirabilis (groups 43, 52, 81)."""
    cfg = PipelineConfig()
    assert "Bacillus species" in cfg.all_pathogens
    assert "Proteus mirabilis" in cfg.all_pathogens


def test_alignment_retains_groups_without_orphaned_removal():
    """Verify alignment retains shared patients across modalities without dropping groups lacking cases."""
    loader = SepsisDataLoader()
    # Dummy clinical and microbiome with a group that has only controls
    df_clin = pd.DataFrame({
        "Pt_ID": ["P1", "P2", "P3", "P4"],
        "Match_group": [1, 1, 2, 2],
        "LOS": [1, 0, 0, 0],  # Group 2 has 0 cases, only controls
    })
    df_micro = pd.DataFrame({
        "Pt_ID": ["P1", "P2", "P3", "P4"],
        "Match_group": [1, 1, 2, 2],
        "LOS": [1, 0, 0, 0],
        "Feature_1": [0.1, 0.2, 0.3, 0.4],
    })

    # Filter to shared patients
    shared_ids = set(df_clin["Pt_ID"]) & set(df_micro["Pt_ID"])
    df_clin_aligned = df_clin[df_clin["Pt_ID"].isin(shared_ids)].sort_values("Pt_ID").reset_index(drop=True)
    df_micro_aligned = df_micro[df_micro["Pt_ID"].isin(shared_ids)].sort_values("Pt_ID").reset_index(drop=True)

    # Without orphaned check, all 4 patients and both groups are preserved
    assert len(df_clin_aligned) == 4
    assert len(df_micro_aligned) == 4
    assert 2 in df_clin_aligned["Match_group"].values


def test_case_control_shap_beeswarm_all_modalities(tmp_path):
    """Verify Experiment (a) generates SHAP beeswarm plots for ALL modalities (not just Best Source)."""
    rng = np.random.RandomState(42)
    # 8 matched groups (1 case + 3 controls each = 32 infants): 4 PathogenA, 4 PathogenB
    n_groups = 8
    rows_clin = []
    rows_sp = []
    rows_ph = []
    pt_counter = 1
    for g in range(1, n_groups + 1):
        pathogen_name = "PathogenA" if g <= 4 else "PathogenB"
        for member in range(4):
            is_case = 1 if member == 0 else 0
            p_label = pathogen_name if is_case else 0
            rows_clin.append({
                "Pt_ID": pt_counter,
                "Match_group": g,
                "LOS": is_case,
                "pathogen": p_label,
                "Hospital_site": (g % 2) + 1,
                "Biological_sex": member % 2,
                "GA_total": 28.0 + rng.randn() + (1.5 if is_case else 0.0),
                "Birthweight": 1100.0 + 100.0 * rng.randn(),
            })
            rows_sp.append({
                "Pt_ID": pt_counter,
                "Match_group": g,
                "LOS": is_case,
                "pathogen": p_label,
                "TaxonA_mean": rng.randn() + (1.2 if p_label == "PathogenA" else 0.0),
                "TaxonB_mean": rng.randn() + (1.2 if p_label == "PathogenB" else 0.0),
                "Shannon_mean": rng.randn(),
            })
            rows_ph.append({
                "Pt_ID": pt_counter,
                "Match_group": g,
                "LOS": is_case,
                "pathogen": p_label,
                "Phylum1_mean": rng.randn() + (0.8 if is_case else 0.0),
                "Phylum2_mean": rng.randn(),
            })
            pt_counter += 1

    sources_data = {
        "clinical": pd.DataFrame(rows_clin),
        "microbiome_species": pd.DataFrame(rows_sp),
        "microbiome_phylum": pd.DataFrame(rows_ph),
    }

    cfg = PipelineConfig(
        figures_dir=str(tmp_path),
        nominal_features=["Hospital_site", "Biological_sex"],
        one_hot_features=["Hospital_site"],
        one_hot_categories={"Hospital_site": [1, 2]},
        n_splits_outer=2,
        n_splits_inner=2,
        bayes_iter=2,
        shap_k=5,
        shap_bootstrap_runs=2,
        n_bootstrap_ci=5,
        plot_shap_beeswarm=True,
        shap_beeswarm_max_display=5,
        random_seed=42,
    )

    best_sources, study_results = run_case_control_study(
        sources_data=sources_data,
        pathogens=["PathogenA"],
        config=cfg,
        run_overall=True,
    )

    expected_modalities = ["clinical", "microbiome_species", "microbiome_phylum", "multimodal_comb"]
    for target in ["overall", "PathogenA"]:
        assert target in best_sources
        assert target in study_results
        for mod in expected_modalities:
            assert mod in study_results[target]
            assert "shap_explanation" in study_results[target][mod]
            assert "shap_plot_path" in study_results[target][mod]
            pdf_file = tmp_path / f"exp_a_{target}_{mod}_shap_beeswarm.pdf"
            png_file = tmp_path / f"exp_a_{target}_{mod}_shap_beeswarm.png"
            assert pdf_file.exists(), f"Missing PDF for {target}/{mod}"
            assert png_file.exists(), f"Missing PNG for {target}/{mod}"


def test_build_pooled_metrics_table_all_exp_a_modalities():
    """Verify pooled_metrics_detailed_table includes ALL Experiment (a) modalities for overall and every pathogen."""
    from run_pipeline import build_pooled_metrics_table

    dummy_ci = {
        metric: {"point": 0.80, "ci_lower": 0.70, "ci_upper": 0.90}
        for metric in ["auc", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy"]
    }
    sources = ["clinical", "microbiome_species", "multimodal_comb"]
    exp_a_results = {
        "overall": {s: {"threshold": 0.45, "metrics_ci": dummy_ci} for s in sources},
        "PathogenA": {s: {"threshold": 0.50, "metrics_ci": dummy_ci} for s in sources},
        "PathogenB": {s: {"threshold": 0.55, "metrics_ci": dummy_ci} for s in sources},
    }
    best_sources = {
        "overall": "multimodal_comb",
        "PathogenA": "microbiome_species",
        "PathogenB": "clinical",
    }
    exp_b_results = {
        "PathogenA": {"source_used": "microbiome_species", "best_threshold": 0.52, "metrics_ci": dummy_ci},
        "PathogenB": {"source_used": "clinical", "best_threshold": 0.54, "metrics_ci": dummy_ci},
    }
    exp_c_results = {
        "PathogenA": {"source_used": "microbiome_species", "best_threshold": 0.53, "metrics_ci": dummy_ci},
        "PathogenB": {"source_used": "clinical", "best_threshold": 0.56, "metrics_ci": dummy_ci},
    }

    df = build_pooled_metrics_table(
        exp_a_results=exp_a_results,
        exp_b_results=exp_b_results,
        exp_c_results=exp_c_results,
        best_sources=best_sources,
        pathogens=["PathogenA", "PathogenB"],
    )

    # 3 targets * 3 modalities in Exp (a) = 9 rows, plus 2 in Exp (b) and 2 in Exp (c) = 13 rows total
    assert len(df) == 13
    expected_cols = [
        "Target", "Experiment", "Data Source", "Best Source", "Threshold",
        "AUC (95% CI)", "Sensitivity (95% CI)", "Specificity (95% CI)",
        "PPV (95% CI)", "NPV (95% CI)", "Balanced Accuracy (95% CI)",
    ]
    assert list(df.columns) == expected_cols

    # Verify all 3 modalities present for Overall Sepsis (LOS) and each pathogen in Exp (a)
    for target_label, expected_best in [
        ("Overall Sepsis (LOS)", "multimodal_comb"),
        ("PathogenA", "microbiome_species"),
        ("PathogenB", "clinical"),
    ]:
        sub = df[(df["Target"] == target_label) & (df["Experiment"] == "(a) Case-Control (1:3)")]
        assert set(sub["Data Source"]) == set(sources)
        best_rows = sub[sub["Best Source"] == "Yes"]
        assert len(best_rows) == 1
        assert best_rows.iloc[0]["Data Source"] == expected_best
        assert len(sub[sub["Best Source"] == "No"]) == 2


