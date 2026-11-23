"""
Experiment execution module.
Implements:
  (a) Case-control (1:3) study across sources and dynamic Best Source identification
  (b) Full cohort test including all controls (excluding other sepsis)
  (c) Full cohort test including all controls and other sepsis
All experiments output pooled held-out test metrics with 100-run bootstrap 95% CIs
and publication-ready SHAP beeswarm summary plots for feature importance in Experiment (a).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from .config import PipelineConfig
from .data_loader import SepsisDataLoader
from .evaluation import (
    bootstrap_pooled_metrics,
    compute_binary_metrics,
    custom_confusion_matrix,
    select_threshold_max_spec_slope,
)
from .feature_selection import stratified_shap_stability_select
from .modeling import train_xgboost_bayes
from .preprocessing import ClinicalSplitPreprocessor
from .visualization import (
    build_union_and_align_shap,
    plot_multi_confusion,
    plot_performance_curves,
    plot_probability_distribution,
    plot_roc_curve,
    plot_shap_beeswarm,
)


def _prepare_source_matrices_split_aware(
    sources_data: Dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: PipelineConfig,
    fold_seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Fit clinical preprocessor strictly on train_idx and transform both train and test.
    Extract microbiome matrices strictly aligned.
    """
    prepared = {}
    meta_cols = {"Pt_ID", "Match_group", "general sepsis", "source sepsis", "pathogen", "LOS"}

    for src_name, df_src in sources_data.items():
        if src_name == "clinical":
            preprocessor = ClinicalSplitPreprocessor(
                nominal_features=config.nominal_features,
                one_hot_features=config.one_hot_features,
                one_hot_categories=getattr(config, "one_hot_categories", None),
                random_state=fold_seed,
            )
            df_train = df_src.iloc[train_idx].copy()
            df_test = df_src.iloc[test_idx].copy()

            preprocessor.fit(df_train)
            X_tr, feat_names = preprocessor.transform(df_train)
            X_te, _ = preprocessor.transform(df_test)
        else:
            feat_cols = sorted([c for c in df_src.columns if c not in meta_cols])
            X_tr = df_src[feat_cols].iloc[train_idx].to_numpy(dtype=float)
            X_te = df_src[feat_cols].iloc[test_idx].to_numpy(dtype=float)
            # Prefix feature names with source identifier
            src_prefix = src_name.split("_")[1] if "_" in src_name else src_name
            feat_names = [
                f"{src_prefix}_{f}" if ("Shannon" in f or "Dominance" in f) else f
                for f in feat_cols
            ]

        prepared[src_name] = (X_tr, X_te, feat_names)

    return prepared


def get_target_aware_outer_cv(
    ref_df: pd.DataFrame,
    target_pathogen: str,
    pathogen_series: np.ndarray,
    group_series: np.ndarray,
    n_splits: int = 5,
    base_seed: int = 33,
    max_search_seeds: int = 50,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate target-aware outer cross-validation folds ensuring balanced positive representation
    for the specific target pathogen across all folds of the full cohort.
    Prevents single-fold clumps and zero-case test folds.
    """
    y_strat = (pathogen_series == target_pathogen).astype(int)
    n_pos = int(y_strat.sum())
    actual_splits = min(n_splits, n_pos) if n_pos >= 2 else n_splits

    best_splits = None
    best_min_pos = -1

    for offset in range(max_search_seeds):
        seed = base_seed + offset
        sgkf = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
        splits = list(sgkf.split(ref_df, y_strat, group_series))
        test_pos_counts = [int(y_strat[te].sum()) for _, te in splits]
        min_pos = min(test_pos_counts) if test_pos_counts else 0

        # Optimal: every test fold has >= 1 positive case
        if min_pos >= 1:
            return splits

        if min_pos > best_min_pos:
            best_min_pos = min_pos
            best_splits = splits

    return best_splits if best_splits is not None else splits


def run_case_control_study(
    sources_data: Dict[str, pd.DataFrame],
    pathogens: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
    run_overall: bool = True,
) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Experiment (a): Case-control (1:3) cross-validation study.
    Evaluates clinical and microbiome modalities (+ multimodal combination).
    Selects the 'Best Source' for each pathogen and overall LOS based on pooled CV AUC.
    Computes pooled held-out metrics with 100-run bootstrap 95% CIs and SHAP beeswarm plots
    for each target and data source.

    Returns:
    --------
    best_sources : Dict[str, str]
        Mapping from pathogen/overall to the best-performing data source name.
    study_results : Dict[str, Dict]
        Detailed metrics, CIs, and pooled predictions for each target and modality.
    """
    cfg = config or PipelineConfig()
    if pathogens is None:
        pathogens = cfg.default_specific_pathogens

    ref_df = sources_data["clinical"]
    group_ref = ref_df["Match_group"].to_numpy()
    pathogen_ref = ref_df["pathogen"].to_numpy()
    pt_ids_ref = ref_df["Pt_ID"].to_numpy()

    targets_to_run = []
    if run_overall:
        targets_to_run.append("overall")
    targets_to_run.extend(pathogens)

    best_sources: Dict[str, str] = {}
    study_results: Dict[str, Dict] = {}

    for target in targets_to_run:
        target_display = "Overall Sepsis (LOS)" if target == "overall" else target.strip()
        print(f"\n[Experiment (a) Case-Control] Running evaluation for: {target_display}")

        # Filter cohort for the target
        if target == "overall":
            target_mask = np.ones(len(ref_df), dtype=bool)
            y_all = ref_df["LOS"].to_numpy(dtype=int)
        else:
            # Matched groups corresponding to target pathogen cases
            matched_groups = np.unique(group_ref[pathogen_ref == target])
            target_mask = np.isin(group_ref, matched_groups)
            y_all = (pathogen_ref == target).astype(int)

        cohort_indices = np.where(target_mask)[0]
        group_cohort = group_ref[cohort_indices]
        y_cohort = y_all[cohort_indices]

        n_pos = int((y_cohort == 1).sum())
        if n_pos < 2:
            print(f"Skipping {target}: insufficient positive cases ({n_pos}).")
            continue

        n_splits = min(cfg.n_splits_outer, n_pos)
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)

        # Track pooled predictions and SHAP explanations for each modality
        modality_predictions = {src: {"y_true": [], "proba": []} for src in sources_data.keys()}
        modality_predictions["multimodal_comb"] = {"y_true": [], "proba": []}
        modality_shap = {src: {"vals": [], "base": [], "data": [], "feats": []} for src in sources_data.keys()}
        modality_shap["multimodal_comb"] = {"vals": [], "base": [], "data": [], "feats": []}

        for fold, (train_fold_idx, test_fold_idx) in enumerate(outer_cv.split(cohort_indices, y_cohort, group_cohort)):
            train_idx = cohort_indices[train_fold_idx]
            test_idx = cohort_indices[test_fold_idx]
            fold_seed = cfg.random_seed + fold * 10

            prepared_sources = _prepare_source_matrices_split_aware(
                sources_data=sources_data,
                train_idx=train_idx,
                test_idx=test_idx,
                config=cfg,
                fold_seed=fold_seed,
            )

            y_tr = y_cohort[train_fold_idx]
            y_te = y_cohort[test_fold_idx]

            selected_features = {}

            # Train each individual source
            for src_name, (X_tr_raw, X_te_raw, feat_names) in prepared_sources.items():
                if X_tr_raw.shape[1] > cfg.shap_k:
                    col_sel = stratified_shap_stability_select(
                        X=X_tr_raw,
                        y=y_tr,
                        n_features_to_select=cfg.shap_k,
                        n_bootstrap=cfg.shap_bootstrap_runs,
                        sample_fraction=cfg.shap_sample_fraction,
                        random_state=fold_seed,
                        adaptive_cap=cfg.adaptive_feature_selection,
                        feature_names=feat_names,
                        target_pathogen=target if target != "overall" else None,
                        prioritize_causative=True,
                    )
                else:
                    col_sel = np.arange(X_tr_raw.shape[1])

                selected_features[src_name] = col_sel
                X_tr_sel = X_tr_raw[:, col_sel]
                X_te_sel = X_te_raw[:, col_sel]

                model, _ = train_xgboost_bayes(
                    X_train=X_tr_sel,
                    y_train=y_tr,
                    groups=group_ref[train_idx],
                    config=cfg,
                    random_state=fold_seed,
                )

                proba_te = model.predict_proba(X_te_sel)[:, 1]
                modality_predictions[src_name]["proba"].extend(proba_te)
                modality_predictions[src_name]["y_true"].extend(y_te)

                # Compute SHAP explanations for this source on held-out fold
                if cfg.plot_shap_beeswarm:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_exp = explainer(X_te_sel)
                        modality_shap[src_name]["vals"].append(shap_exp.values)
                        modality_shap[src_name]["base"].append(shap_exp.base_values)
                        modality_shap[src_name]["data"].append(shap_exp.data)
                        modality_shap[src_name]["feats"].append(list(np.array(feat_names)[col_sel]))
                    except Exception:
                        pass

            # Multimodal combination: concatenate selected features from all modalities
            comb_tr = np.column_stack([
                prepared_sources[s][0][:, selected_features[s]] for s in sources_data.keys()
            ])
            comb_te = np.column_stack([
                prepared_sources[s][1][:, selected_features[s]] for s in sources_data.keys()
            ])
            comb_feat_names = np.concatenate([
                np.array(prepared_sources[s][2])[selected_features[s]] for s in sources_data.keys()
            ])

            if comb_tr.shape[1] > cfg.shap_k:
                col_sel_comb = stratified_shap_stability_select(
                    X=comb_tr,
                    y=y_tr,
                    n_features_to_select=cfg.shap_k,
                    n_bootstrap=cfg.shap_bootstrap_runs,
                    sample_fraction=cfg.shap_sample_fraction,
                    random_state=fold_seed,
                    adaptive_cap=cfg.adaptive_feature_selection,
                    feature_names=comb_feat_names,
                    target_pathogen=target if target != "overall" else None,
                    prioritize_causative=True,
                )
            else:
                col_sel_comb = np.arange(comb_tr.shape[1])

            model_comb, _ = train_xgboost_bayes(
                X_train=comb_tr[:, col_sel_comb],
                y_train=y_tr,
                groups=group_ref[train_idx],
                config=cfg,
                random_state=fold_seed,
            )
            proba_te_comb = model_comb.predict_proba(comb_te[:, col_sel_comb])[:, 1]
            modality_predictions["multimodal_comb"]["proba"].extend(proba_te_comb)
            modality_predictions["multimodal_comb"]["y_true"].extend(y_te)

            if cfg.plot_shap_beeswarm:
                try:
                    explainer_comb = shap.TreeExplainer(model_comb)
                    shap_exp_comb = explainer_comb(comb_te[:, col_sel_comb])
                    modality_shap["multimodal_comb"]["vals"].append(shap_exp_comb.values)
                    modality_shap["multimodal_comb"]["base"].append(shap_exp_comb.base_values)
                    modality_shap["multimodal_comb"]["data"].append(shap_exp_comb.data)
                    modality_shap["multimodal_comb"]["feats"].append(list(comb_feat_names[col_sel_comb]))
                except Exception:
                    pass

        # Evaluate all modalities for this target
        target_results = {}
        best_auc = -1.0
        best_src = None

        for src_name, preds in modality_predictions.items():
            y_true_arr = np.array(preds["y_true"], dtype=int)
            prob_arr = np.array(preds["proba"], dtype=float)

            thresh, _ = select_threshold_max_spec_slope(y_true_arr, prob_arr)
            ci_metrics = bootstrap_pooled_metrics(
                y_true=y_true_arr,
                probabilities=prob_arr,
                threshold=thresh,
                n_bootstrap=cfg.n_bootstrap_ci,
                ci_level=cfg.ci_level,
                random_state=cfg.random_seed,
            )

            target_results[src_name] = {
                "threshold": thresh,
                "metrics_ci": ci_metrics,
                "y_true": y_true_arr,
                "probabilities": prob_arr,
            }

            auc_point = ci_metrics["auc"]["point"]
            if np.isfinite(auc_point) and auc_point > best_auc:
                best_auc = auc_point
                best_src = src_name

        best_sources[target] = best_src or "clinical"
        study_results[target] = target_results
        print(f" -> Best source for {target_display}: {best_sources[target]} (AUC: {best_auc:.3f})")

        # Generate and save SHAP Beeswarm summary plots for each evaluated source (not only the best source)
        if cfg.plot_shap_beeswarm:
            target_slug = "overall" if target == "overall" else target.strip()
            chosen_src = best_sources[target]
            for src_name, shap_data in modality_shap.items():
                if len(shap_data["vals"]) > 0:
                    try:
                        aligned_exp = build_union_and_align_shap(
                            shap_values_list=shap_data["vals"],
                            shap_base_list=shap_data["base"],
                            shap_data_list=shap_data["data"],
                            shap_featnames_list=shap_data["feats"],
                        )
                        if src_name in target_results:
                            target_results[src_name]["shap_explanation"] = aligned_exp

                        save_path = f"{cfg.figures_dir}/exp_a_{target_slug}_{src_name}_shap_beeswarm.pdf"
                        title = f"SHAP Feature Importance (Beeswarm) - {target_display}\nExperiment (a) Case-Control [Source: {src_name}]"
                        plot_shap_beeswarm(
                            explanation=aligned_exp,
                            title=title,
                            save_path=save_path,
                            max_display=cfg.shap_beeswarm_max_display,
                        )
                        if src_name in target_results:
                            target_results[src_name]["shap_plot_path"] = save_path
                        print(f" -> Saved SHAP beeswarm plot: {save_path}")

                        # Also save unadorned filename for best source for backward compatibility
                        if src_name == chosen_src:
                            best_save_path = f"{cfg.figures_dir}/exp_a_{target_slug}_shap_beeswarm.pdf"
                            plot_shap_beeswarm(
                                explanation=aligned_exp,
                                title=title,
                                save_path=best_save_path,
                                max_display=cfg.shap_beeswarm_max_display,
                            )
                    except Exception as e:
                        print(f" -> [Warning] Could not generate SHAP beeswarm for {target_display} [{src_name}]: {e}")

    return best_sources, study_results


def run_full_cohort_study(
    sources_data: Dict[str, pd.DataFrame],
    best_sources: Dict[str, str],
    pathogens: Optional[List[str]] = None,
    include_other_sepsis: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict]:
    """
    Run full cohort evaluation for target pathogens.
    Trains on target pathogen cases vs matched controls (1:3).
    Evaluates on:
      - (b) All controls (excluding other sepsis) if include_other_sepsis is False.
      - (c) All controls + other sepsis if include_other_sepsis is True.
    """
    cfg = config or PipelineConfig()
    if pathogens is None:
        pathogens = cfg.default_specific_pathogens

    ref_df = sources_data["clinical"]
    group_ref = ref_df["Match_group"].to_numpy()
    pathogen_ref = ref_df["pathogen"].to_numpy()
    pt_ids_ref = ref_df["Pt_ID"].to_numpy()

    exp_label = "(c) Full Cohort + Other Sepsis" if include_other_sepsis else "(b) Full Cohort All Controls"
    print(f"\n{'='*70}\nStarting Experiment {exp_label}\n{'='*70}")

    pathogen_predictions = {
        p: {"true_labels": [], "probabilities": [], "pt_ids": [], "pna": []}
        for p in pathogens
    }

    # Extract PNA from clinical source for heatmap visualization
    if "Postnatal_age_sepsis" in ref_df.columns:
        pna_series = ref_df["Postnatal_age_sepsis"].to_numpy(dtype=float)
    else:
        pna_series = np.full(len(ref_df), np.nan)

    for target_pathogen in pathogens:
        p_name_clean = target_pathogen.strip()
        print(f"\n[Experiment {exp_label}] Running target-aware evaluation for: {p_name_clean}")

        # Target-aware outer CV splits for this specific pathogen across the full cohort
        outer_splits = get_target_aware_outer_cv(
            ref_df=ref_df,
            target_pathogen=target_pathogen,
            pathogen_series=pathogen_ref,
            group_series=group_ref,
            n_splits=cfg.n_splits_outer,
            base_seed=cfg.random_seed,
        )

        for fold, (train_idx, test_idx) in enumerate(outer_splits):
            fold_seed = cfg.random_seed + fold * 10

            prepared_sources = _prepare_source_matrices_split_aware(
                sources_data=sources_data,
                train_idx=train_idx,
                test_idx=test_idx,
                config=cfg,
                fold_seed=fold_seed,
            )

            # 1. Identify training subset: Target pathogen cases vs matched controls (1:3)
            matched_groups_in_train = np.unique(group_ref[train_idx][pathogen_ref[train_idx] == target_pathogen])
            in_target_group_train = np.isin(group_ref[train_idx], matched_groups_in_train)
            train_idx_target = train_idx[in_target_group_train]
            y_train_target = (pathogen_ref[train_idx_target] == target_pathogen).astype(int)

            n_target_pos = int((y_train_target == 1).sum())
            if n_target_pos == 0:
                continue

            # 2. Select data source according to Experiment (a) Best Source
            src_selected = best_sources.get(target_pathogen, best_sources.get("overall", "clinical"))

            if src_selected == "multimodal_comb":
                # Combine selected features across all modalities
                X_tr_full = np.column_stack([prepared_sources[s][0] for s in sources_data.keys()])
                X_te_full = np.column_stack([prepared_sources[s][1] for s in sources_data.keys()])
                feat_names_selected = np.concatenate([prepared_sources[s][2] for s in sources_data.keys()])
            else:
                X_tr_full = prepared_sources[src_selected][0]
                X_te_full = prepared_sources[src_selected][1]
                feat_names_selected = prepared_sources[src_selected][2]

            # Filter training data to target pathogen matched cohort
            X_train_target = X_tr_full[in_target_group_train]

            # Feature selection fitted strictly on training split
            if X_train_target.shape[1] > cfg.shap_k:
                col_sel = stratified_shap_stability_select(
                    X=X_train_target,
                    y=y_train_target,
                    n_features_to_select=cfg.shap_k,
                    n_bootstrap=cfg.shap_bootstrap_runs,
                    sample_fraction=cfg.shap_sample_fraction,
                    random_state=fold_seed,
                    adaptive_cap=cfg.adaptive_feature_selection,
                    feature_names=feat_names_selected,
                    target_pathogen=target_pathogen,
                    prioritize_causative=True,
                )
            else:
                col_sel = np.arange(X_train_target.shape[1])

            X_tr_sel = X_train_target[:, col_sel]
            X_te_sel = X_te_full[:, col_sel]

            # Train classifier
            model, _ = train_xgboost_bayes(
                X_train=X_tr_sel,
                y_train=y_train_target,
                groups=group_ref[train_idx_target],
                config=cfg,
                random_state=fold_seed,
            )

            proba_test = model.predict_proba(X_te_sel)[:, 1]

            # Assign true multi-class labels for evaluation:
            # 0: Target pathogen
            # 1: Other sepsis
            # 2: Control (all matched controls from all groups)
            fold_true_labels = []
            for pid in pt_ids_ref[test_idx]:
                p_label = pathogen_ref[pt_ids_ref == pid][0]
                if p_label == 0 or p_label == "0":
                    true_class = 2  # Control
                elif p_label == target_pathogen:
                    true_class = 0  # Target Pathogen
                else:
                    true_class = 1  # Other sepsis
                fold_true_labels.append(true_class)

            pathogen_predictions[target_pathogen]["true_labels"].extend(fold_true_labels)
            pathogen_predictions[target_pathogen]["probabilities"].extend(proba_test)
            pathogen_predictions[target_pathogen]["pt_ids"].extend(pt_ids_ref[test_idx])
            pathogen_predictions[target_pathogen]["pna"].extend(pna_series[test_idx])

    # Filter out other sepsis if include_other_sepsis is False (Experiment b)
    if not include_other_sepsis:
        for p in pathogens:
            filtered = [
                (l, pr, pid, pna)
                for l, pr, pid, pna in zip(
                    pathogen_predictions[p]["true_labels"],
                    pathogen_predictions[p]["probabilities"],
                    pathogen_predictions[p]["pt_ids"],
                    pathogen_predictions[p]["pna"],
                )
                if l != 1
            ]
            if filtered:
                pathogen_predictions[p]["true_labels"] = [f[0] for f in filtered]
                pathogen_predictions[p]["probabilities"] = [f[1] for f in filtered]
                pathogen_predictions[p]["pt_ids"] = [f[2] for f in filtered]
                pathogen_predictions[p]["pna"] = [f[3] for f in filtered]

    # Threshold optimization, metrics, bootstrap CI, and figures
    experiment_results = {}
    for p in pathogens:
        p_data = pathogen_predictions[p]
        if len(p_data["true_labels"]) == 0:
            continue

        raw_labels = np.array(p_data["true_labels"], dtype=int)
        raw_probs = np.array(p_data["probabilities"], dtype=float)

        # Binary conversion: Target pathogen (0) is positive (1), all other tested are negative (0)
        y_binary = (raw_labels == 0).astype(int)

        if len(np.unique(y_binary)) < 2:
            print(f"Skipping evaluation for {p}: no positive cases in test cohort.")
            continue

        # Keep threshold optimization on test set for unique threshold
        best_threshold, point_metrics = select_threshold_max_spec_slope(y_binary, raw_probs)

        # 100-run bootstrap confidence intervals
        ci_summary = bootstrap_pooled_metrics(
            y_true=y_binary,
            probabilities=raw_probs,
            threshold=best_threshold,
            n_bootstrap=cfg.n_bootstrap_ci,
            ci_level=cfg.ci_level,
            random_state=cfg.random_seed,
        )

        # Visualizations
        p_name_clean = p.strip()
        figures_prefix = f"{cfg.figures_dir}/exp_{'c' if include_other_sepsis else 'b'}_{p_name_clean}"

        plot_probability_distribution(
            pathogen_data=p_data,
            pathogen_name=p_name_clean,
            include_other_sepsis=include_other_sepsis,
            save_path=f"{figures_prefix}_prob_distribution.pdf",
        )
        plot_performance_curves(
            pathogen_data=p_data,
            pathogen_name=p_name_clean,
            save_path=f"{figures_prefix}_performance_curves.pdf",
        )
        plot_roc_curve(
            y_true=y_binary,
            probabilities=raw_probs,
            pathogen_name=p_name_clean,
            save_path=f"{figures_prefix}_roc_curve.pdf",
            ci_info=ci_summary["auc"],
        )

        y_pred_binary = (raw_probs >= best_threshold).astype(int)
        cm = custom_confusion_matrix(raw_labels, y_pred_binary, include_other_sepsis=include_other_sepsis)
        plot_multi_confusion(
            cm=cm,
            title=f"Confusion Matrix {p_name_clean} (thr={best_threshold:.2f})",
            save_path=f"{figures_prefix}_confusion_matrix.pdf",
            include_other_sepsis=include_other_sepsis,
        )

        experiment_results[p] = {
            "source_used": best_sources.get(p, "clinical"),
            "best_threshold": best_threshold,
            "metrics_ci": ci_summary,
            "confusion_matrix": cm,
            "n_samples": len(raw_labels),
            "n_positives": int((raw_labels == 0).sum()),
        }

        auc_info = ci_summary["auc"]
        sens_info = ci_summary["sensitivity"]
        spec_info = ci_summary["specificity"]
        ppv_info = ci_summary["ppv"]
        npv_info = ci_summary["npv"]

        print(f"\nResults for {p_name_clean} [Source: {best_sources.get(p, 'clinical')}]:")
        print(f"  Threshold:   {best_threshold:.3f}")
        print(f"  AUC:         {auc_info['point']:.3f} [{auc_info['ci_lower']:.3f} - {auc_info['ci_upper']:.3f}]")
        print(f"  Sensitivity: {sens_info['point']:.3f} [{sens_info['ci_lower']:.3f} - {sens_info['ci_upper']:.3f}]")
        print(f"  Specificity: {spec_info['point']:.3f} [{spec_info['ci_lower']:.3f} - {spec_info['ci_upper']:.3f}]")
        print(f"  PPV:         {ppv_info['point']:.3f} [{ppv_info['ci_lower']:.3f} - {ppv_info['ci_upper']:.3f}]")
        print(f"  NPV:         {npv_info['point']:.3f} [{npv_info['ci_lower']:.3f} - {npv_info['ci_upper']:.3f}]")

    return experiment_results
