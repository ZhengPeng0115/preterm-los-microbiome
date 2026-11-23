"""
Evaluation and bootstrap metrics module.
Provides binary performance metrics, test-set threshold optimization,
and pooled 100-run bootstrap confidence interval estimation (95% CI).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (1: positive/pathogen, 0: negative/control)
    y_pred : np.ndarray
        Predicted binary labels (1: positive/pathogen, 0: negative/control)

    Returns:
    --------
    dict
        Dictionary containing sensitivity, specificity, ppv, npv,
        balanced_accuracy, accuracy, kappa, TP, TN, FP, FN.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = float(TP / (TP + FN)) if (TP + FN) > 0 else np.nan
    specificity = float(TN / (TN + FP)) if (TN + FP) > 0 else np.nan
    ppv = float(TP / (TP + FP)) if (TP + FP) > 0 else np.nan
    npv = float(TN / (TN + FN)) if (TN + FN) > 0 else np.nan

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else np.nan
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else np.nan
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "kappa": kappa,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
    }


def select_threshold_max_spec_slope(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Select decision threshold by maximizing the specificity slope within a narrow
    window preceding the Sensitivity-Specificity intersection on the test set.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (1: pathogen, 0: control/other)
    probabilities : np.ndarray
        Predicted probabilities for the positive class.
    thresholds : np.ndarray, optional
        Threshold grid. Defaults to np.linspace(0, 1, 101).

    Returns:
    --------
    Tuple[float, Dict[str, float]]
        (best_threshold, metrics_dict)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    sensitivities = []
    specificities = []

    for thresh in thresholds:
        y_pred = (probabilities >= thresh).astype(int)
        m = compute_binary_metrics(y_true, y_pred)
        sensitivities.append(m["sensitivity"])
        specificities.append(m["specificity"])

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    # Intersection point (where curves are closest)
    diff = np.abs(sensitivities - specificities)
    idx_intersection = int(np.nanargmin(diff))
    thresh_intersection = float(thresholds[idx_intersection])

    # Search window [thresh_intersection - 0.05, thresh_intersection]
    min_thresh = thresh_intersection - 0.05
    best_slope = -np.inf
    best_idx = idx_intersection

    for j in range(1, idx_intersection + 1):
        if thresholds[j] >= min_thresh:
            slope = specificities[j] - specificities[j - 1]
            if slope > best_slope:
                best_slope = slope
                best_idx = j

    best_threshold = float(thresholds[best_idx])
    final_pred = (probabilities >= best_threshold).astype(int)
    final_metrics = compute_binary_metrics(y_true, final_pred)

    try:
        final_metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        final_metrics["auc"] = np.nan

    return best_threshold, final_metrics


def bootstrap_pooled_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 100,
    ci_level: float = 0.95,
    random_state: int = 33,
) -> Dict[str, Dict[str, float]]:
    """
    Compute point estimates and 95% confidence intervals via bootstrapping (100 runs)
    on pooled cross-validation test predictions.

    Returns:
    --------
    dict
        Dictionary mapping metric names to {'point': float, 'mean': float, 'std': float, 'ci_lower': float, 'ci_upper': float}
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_samples = len(y_true)

    # Point estimates on pooled dataset
    y_pred_point = (probabilities >= threshold).astype(int)
    point_metrics = compute_binary_metrics(y_true, y_pred_point)
    try:
        point_metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        point_metrics["auc"] = np.nan

    metric_keys = ["auc", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
    bootstrap_results = {k: [] for k in metric_keys}

    rng = np.random.RandomState(random_state)
    alpha = (1.0 - ci_level) / 2.0

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    for b in range(n_bootstrap):
        # Stratified bootstrap draw to ensure class representation
        if len(idx_pos) > 0 and len(idx_neg) > 0:
            sample_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            sample_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            boot_idx = np.concatenate([sample_pos, sample_neg])
        else:
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)

        y_boot = y_true[boot_idx]
        prob_boot = probabilities[boot_idx]
        pred_boot = (prob_boot >= threshold).astype(int)

        m_boot = compute_binary_metrics(y_boot, pred_boot)
        try:
            auc_boot = float(roc_auc_score(y_boot, prob_boot))
        except ValueError:
            auc_boot = np.nan
        m_boot["auc"] = auc_boot

        for k in metric_keys:
            val = m_boot.get(k, np.nan)
            if np.isfinite(val):
                bootstrap_results[k].append(val)

    # Aggregate CI
    ci_summary = {}
    for k in metric_keys:
        vals = np.array(bootstrap_results[k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            ci_lower = float(np.percentile(vals, alpha * 100.0))
            ci_upper = float(np.percentile(vals, (1.0 - alpha) * 100.0))
            boot_mean = float(np.mean(vals))
            boot_std = float(np.std(vals))
        else:
            ci_lower = np.nan
            ci_upper = np.nan
            boot_mean = np.nan
            boot_std = np.nan

        ci_summary[k] = {
            "point": point_metrics.get(k, np.nan),
            "mean": boot_mean,
            "std": boot_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return ci_summary


def custom_confusion_matrix(
    y_true_categories: np.ndarray,
    y_pred_binary: np.ndarray,
    include_other_sepsis: bool = True,
) -> np.ndarray:
    """
    Construct multi-class confusion matrix.
    Rows: True class (0: Pathogen, 1: Other sepsis [if included], 2: Control).
    Cols: Predicted class (0: Pathogen [pred=1], 1: Control/Other [pred=0]).
    """
    if include_other_sepsis:
        cm = np.zeros((3, 2), dtype=int)
        for t, p in zip(y_true_categories, y_pred_binary):
            # p=1 (pathogen) -> col 0; p=0 (control) -> col 1
            col = 0 if p == 1 else 1
            if t in [0, 1, 2]:
                cm[t, col] += 1
    else:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true_categories, y_pred_binary):
            col = 0 if p == 1 else 1
            if t == 0:  # Pathogen
                cm[0, col] += 1
            elif t == 2:  # Control
                cm[1, col] += 1

    return cm
