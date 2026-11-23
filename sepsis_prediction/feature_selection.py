"""
Feature selection module.
Implements stratified SHAP stability selection with optional adaptive feature capping
and causative pathogen prioritized feature retention.
"""

import sys
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import shap
from xgboost import XGBClassifier


def get_causative_tokens(pathogen_name: str) -> Tuple[str, List[str]]:
    """
    Return (genus_token, [species_tokens]) for a given pathogen name.
    Handles clinical abbreviations and complex species complexes.
    """
    p = pathogen_name.strip().lower()
    mapping = {
        "serratia marcecens": ("serratia", ["marce", "marcescens"]),
        "serratia marcescens": ("serratia", ["marce", "marcescens"]),
        "escherichia coli": ("escherichia", ["coli", "shigella"]),
        "klebsiella spp.": ("klebsiella", []),
        "klebsiella spp. ": ("klebsiella", []),
        "enterobacter cloacae": ("enterobacter", ["cloacae"]),
        "staphylococcus aureus": ("staphylococcus", ["aureus"]),
        "bacillus cereus": ("bacillus", ["cereus"]),
        "enterococcus faecalis": ("enterococcus", ["faecalis"]),
        "group b streptococcus": ("streptococcus", ["group b", "agalactiae"]),
        "pantoea septica": ("pantoea", ["septica"]),
        "pseudomonas spp.": ("pseudomonas", []),
    }
    if p in mapping:
        return mapping[p]

    parts = p.replace(".", "").split()
    genus = parts[0] if parts else ""
    species = parts[1:] if len(parts) > 1 else []
    return genus, species


def find_causative_feature_indices(
    feature_names: Sequence[str],
    target_pathogen: Optional[str],
) -> List[int]:
    """
    Identify column indices in feature_names that represent the causative pathogen.
    Prioritizes specific species-level matches over broader genus-level matches.
    """
    if not target_pathogen or target_pathogen == "overall" or feature_names is None:
        return []

    genus, species_tokens = get_causative_tokens(target_pathogen)
    if not genus:
        return []

    level1_species = []
    level2_genus = []

    for idx, f in enumerate(feature_names):
        f_lower = str(f).lower()
        if genus in f_lower:
            if species_tokens and any(st in f_lower for st in species_tokens):
                level1_species.append(idx)
            else:
                level2_genus.append(idx)
        elif species_tokens and any(st in f_lower for st in species_tokens):
            level1_species.append(idx)

    # Return species-level matches if available, otherwise genus-level
    return level1_species if level1_species else level2_genus


def stratified_shap_stability_select(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 30,
    n_bootstrap: int = 30,
    sample_fraction: float = 0.8,
    random_state: int = 22,
    adaptive_cap: bool = False,
    feature_names: Optional[Sequence[str]] = None,
    target_pathogen: Optional[str] = None,
    prioritize_causative: bool = True,
) -> np.ndarray:
    """
    Select top features using bootstrap-averaged mean(|SHAP|) values with stratified sampling
    and causative pathogen prioritized feature retention.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix [n_samples, n_features]
    y : np.ndarray
        Binary labels (0: control/negative, 1: case/positive)
    n_features_to_select : int
        Maximum number of features to select (default: 30).
    n_bootstrap : int
        Number of bootstrap iterations.
    sample_fraction : float
        Fraction of samples to draw in each bootstrap.
    random_state : int
        Random seed for reproducibility.
    adaptive_cap : bool, default=False
        If True, caps candidate features based on positive event count.
        If False (default), selects consistent n_features_to_select (e.g. 30).
    feature_names : Sequence[str], optional
        List/array of feature names aligned with columns of X.
    target_pathogen : str, optional
        Target pathogen name for prioritizing causative taxonomic features.
    prioritize_causative : bool
        If True, prioritizes retention of causative pathogen features.

    Returns:
    --------
    np.ndarray
        Indices of selected features.
    """
    rng = np.random.RandomState(random_state)
    n, p = X.shape

    if p <= 1:
        return np.arange(p)

    # 1. Feature capping: consistent n_features_to_select (e.g. 30) unless adaptive_cap=True
    n_positives = int((y == 1).sum())
    if adaptive_cap and n_positives > 0:
        max_allowed_features = max(1, int(np.ceil(n_positives * 0.6)))
        k_to_select = min(n_features_to_select, max_allowed_features, p)
    else:
        k_to_select = min(n_features_to_select, p)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    # Handle degenerate case where no positives exist
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return np.arange(k_to_select)

    n_pos_draw = max(1, int(round(sample_fraction * len(idx_pos))))
    n_neg_draw = max(1, int(round(sample_fraction * len(idx_neg))))

    # Single-thread workers when debugger is attached to avoid BrokenProcessPool crashes
    is_debugging = hasattr(sys, "gettrace") and sys.gettrace() is not None
    parallel_jobs = 1 if is_debugging else -1

    # Regularized base XGBoost model to prevent memorization on small bootstrap draws
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=parallel_jobs,
        random_state=random_state,
    )

    shap_sum = np.zeros(p, dtype=float)
    valid_bootstraps = 0

    for b in range(n_bootstrap):
        # Stratified sampling with replacement ensures both positive and negative cases are present
        sample_pos = rng.choice(idx_pos, size=n_pos_draw, replace=True)
        sample_neg = rng.choice(idx_neg, size=n_neg_draw, replace=True)
        idx_b = np.concatenate([sample_pos, sample_neg])
        rng.shuffle(idx_b)

        Xb, yb = X[idx_b], y[idx_b]

        # Verify class presence
        if len(np.unique(yb)) < 2:
            continue

        try:
            base_model.fit(Xb, yb)
            explainer = shap.TreeExplainer(base_model, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(Xb)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            vals = np.abs(shap_vals)
            shap_sum += vals.mean(axis=0)
            valid_bootstraps += 1
        except Exception:
            continue

    if valid_bootstraps == 0:
        variances = np.var(X, axis=0)
        shap_mean = variances
    else:
        shap_mean = shap_sum / valid_bootstraps

    ranked_by_shap = np.argsort(shap_mean)[::-1]

    # Causative Pathogen Prioritized Feature Retention
    if prioritize_causative and feature_names is not None and target_pathogen is not None:
        causative_indices = find_causative_feature_indices(feature_names, target_pathogen)
        if causative_indices:
            # Sort causative features by SHAP importance, then by variance
            causative_sorted = sorted(
                causative_indices,
                key=lambda idx: (shap_mean[idx], np.var(X[:, idx])),
                reverse=True
            )
            # Reserve up to k_causative slots (at least 1, up to max(1, k_to_select // 2))
            k_causative = min(len(causative_sorted), max(1, k_to_select // 2))
            top_causative = causative_sorted[:k_causative]

            # Fill remaining slots with the highest-ranking other features
            remaining_slots = k_to_select - len(top_causative)
            non_causative = [idx for idx in ranked_by_shap if idx not in top_causative]
            top_others = non_causative[:remaining_slots]

            keep_idx = np.array(top_causative + top_others, dtype=int)
            return keep_idx

    keep_idx = ranked_by_shap[:k_to_select]
    return keep_idx
