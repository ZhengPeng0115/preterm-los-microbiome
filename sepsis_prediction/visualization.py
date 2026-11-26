"""
Visualization module for sepsis prediction evaluation.
Provides crash-safe probability distributions, ROC curves,
performance threshold sweeps, multi-class confusion matrices,
and SHAP beeswarm summary plots for feature importance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import roc_auc_score, roc_curve

from .evaluation import compute_binary_metrics


def plot_probability_distribution(
    pathogen_data: Dict[str, List],
    pathogen_name: str,
    include_other_sepsis: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot predicted probability distributions by true label class.
    Crash-safe for small or single-sample classes by falling back to scatter points.
    """
    true_labels = np.array(pathogen_data["true_labels"])
    probabilities = np.array(pathogen_data["probabilities"])

    if include_other_sepsis:
        label_names = [f"{pathogen_name}", "Other sepsis", "Control"]
        label_values = [0, 1, 2]
    else:
        label_names = [f"{pathogen_name}", "Control"]
        label_values = [0, 2]

    data_by_label = []
    means = []
    stds = []
    counts = []

    for label in label_values:
        mask = true_labels == label
        if mask.sum() > 0:
            probs = probabilities[mask]
            data_by_label.append(probs)
            means.append(float(probs.mean()))
            stds.append(float(probs.std()))
            counts.append(int(mask.sum()))
        else:
            data_by_label.append(np.array([]))
            means.append(0.0)
            stds.append(0.0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e74c3c", "#f39c12"] if include_other_sepsis else ["#2ecc71", "#f39c12"]

    # Crash-safe violin plot: KDE requires at least 2 distinct points with non-zero variance
    for i, (data, color) in enumerate(zip(data_by_label, colors)):
        x_pos = i + 1
        if len(data) >= 2 and np.var(data) > 1e-6:
            parts = ax.violinplot(data, positions=[x_pos], showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.6)
        elif len(data) > 0:
            # Fallback scatter for small/constant classes
            jitter = np.random.uniform(-0.05, 0.05, size=len(data))
            ax.scatter(np.full_like(data, x_pos) + jitter, data, color=color, alpha=0.8, edgecolor="black", s=50)

    # Add boxplot overlay
    valid_data = [d if len(d) > 0 else [0.0] for d in data_by_label]
    bp = ax.boxplot(valid_data, positions=np.arange(1, len(label_names) + 1), widths=0.15,
                    patch_artist=True, showfliers=False, zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)

    # Scatter points with jitter
    rng = np.random.RandomState(42)
    for i, (data, color) in enumerate(zip(data_by_label, colors)):
        if len(data) > 0:
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(np.full_like(data, i + 1) + jitter, data, color=color, alpha=0.4,
                       edgecolors="black", linewidth=0.5, s=35, zorder=4)

    # Annotations
    for i, (count, mean, std) in enumerate(zip(counts, means, stds)):
        ax.text(i + 1, 1.05, f"n={count}\n{mean:.3f}±{std:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_ylabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_xlabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title(f"Predicted Probability Distribution\n{pathogen_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(1, len(label_names) + 1))
    ax.set_xticklabels(label_names, fontsize=10)
    ax.set_ylim(-0.05, 1.2)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_performance_curves(
    pathogen_data: Dict[str, List],
    pathogen_name: str,
    thresholds: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot Sensitivity, Specificity, PPV, and NPV across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    true_labels = np.array(pathogen_data["true_labels"])
    probabilities = np.array(pathogen_data["probabilities"])
    y_true = (true_labels == 0).astype(int)

    metrics = {"Sensitivity": [], "Specificity": [], "PPV": [], "NPV": []}
    for thresh in thresholds:
        y_pred = (probabilities >= thresh).astype(int)
        m = compute_binary_metrics(y_true, y_pred)
        metrics["Sensitivity"].append(m["sensitivity"])
        metrics["Specificity"].append(m["specificity"])
        metrics["PPV"].append(m["ppv"])
        metrics["NPV"].append(m["npv"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in metrics.items():
        ax.plot(thresholds, values, label=name, linewidth=2)

    ax.set_title(f"Performance vs Threshold\n{pathogen_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="best")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    pathogen_name: str,
    save_path: Optional[Union[str, Path]] = None,
    ci_info: Optional[Dict[str, float]] = None,
) -> None:
    """Plot ROC curve with AUC and confidence intervals."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    if ci_info and "ci_lower" in ci_info and "ci_upper" in ci_info:
        label = f"{pathogen_name} (AUC = {auc:.2f} [{ci_info['ci_lower']:.2f} - {ci_info['ci_upper']:.2f}])"
    else:
        label = f"{pathogen_name} (AUC = {auc:.2f})"

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=label)
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(f"ROC Curve - {pathogen_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_multi_confusion(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    include_other_sepsis: bool = True,
) -> None:
    """Plot multi-class or binary confusion matrix with clean annotations."""
    if include_other_sepsis:
        true_labels = ("Pathogen", "Other sepsis", "Control")
    else:
        true_labels = ("Pathogen", "Control")
    pred_labels = ("Pathogen", "Control")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=pred_labels,
        yticklabels=true_labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_auc_heatmap(
    pathogen_data: Dict[str, List],
    pathogen_name: str,
    min_diff: int = 5,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot AUC vs Postnatal Age (PNA) heatmap."""
    true_labels = np.array(pathogen_data["true_labels"])
    probabilities = np.array(pathogen_data["probabilities"])
    pna = np.array(pathogen_data.get("pna", []))
    y_true = (true_labels == 0).astype(int)

    valid_mask = ~np.isnan(pna)
    pna = pna[valid_mask]
    y_true = y_true[valid_mask]
    probabilities = probabilities[valid_mask]

    if len(pna) < 5 or len(np.unique(y_true)) < 2:
        return

    pna_int = np.round(pna).astype(int)
    min_pna = int(pna_int.min())
    max_pna = int(pna_int.max())

    unique_pna = np.arange(min_pna, max_pna + 1)
    n_pna = len(unique_pna)

    scores = np.full((n_pna, n_pna), np.nan)
    positives = np.zeros((n_pna, n_pna), dtype=int)
    negatives = np.zeros((n_pna, n_pna), dtype=int)

    for i, l_limit in enumerate(unique_pna):
        for j, u_limit in enumerate(unique_pna):
            if u_limit < l_limit + min_diff:
                continue
            mask = (pna_int >= l_limit) & (pna_int <= u_limit)
            if mask.sum() < 2:
                continue
            y_sub = y_true[mask]
            prob_sub = probabilities[mask]
            if len(np.unique(y_sub)) < 2:
                continue
            try:
                scores[j, i] = roc_auc_score(y_sub, prob_sub)
                positives[j, i] = int(y_sub.sum())
                negatives[j, i] = int(len(y_sub) - y_sub.sum())
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(unique_pna, unique_pna)
    mesh = ax.pcolormesh(X, Y, scores, cmap="viridis", shading="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(mesh, label="AUC")

    ax.set_xlabel("Lower PNA Limit (days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Upper PNA Limit (days)", fontsize=12, fontweight="bold")
    ax.set_title(f"AUC vs PNA Range (>={min_diff} days)\n{pathogen_name}", fontsize=14, fontweight="bold")
    ax.set_xlim(min_pna, max_pna - min_diff)
    ax.set_ylim(min_pna + min_diff, max_pna)
    ax.grid(alpha=0.3, linestyle=":")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def build_union_and_align_shap(
    shap_values_list: List[np.ndarray],
    shap_base_list: List[Union[float, np.ndarray]],
    shap_data_list: List[Optional[np.ndarray]],
    shap_featnames_list: List[List[str]],
    full_feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """
    Align fold-specific SHAP explanation matrices across cross-validation splits
    into a unified pooled Explanation object across the union of all selected features.
    Features not present in a given fold receive SHAP values of 0.0 and NaN data.

    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of [n_i, k_i] arrays of SHAP values per fold.
    shap_base_list : List[Union[float, np.ndarray]]
        List of [n_i] or scalar base values (expected values) per fold.
    shap_data_list : List[Optional[np.ndarray]]
        List of [n_i, k_i] feature data matrices per fold, or None.
    shap_featnames_list : List[List[str]]
        List of lists of feature names per fold.
    full_feature_names : Optional[List[str]]
        Optional canonical global ordering of features.

    Returns
    -------
    shap.Explanation
        Pooled Explanation combining all test samples across folds.
    """
    union_set = set()
    for names in shap_featnames_list:
        union_set.update(names)

    if full_feature_names is not None:
        union_names = [f for f in full_feature_names if f in union_set]
    else:
        union_names = sorted(list(union_set))

    name_to_pos = {name: j for j, name in enumerate(union_names)}
    p_union = len(union_names)

    aligned_vals_list = []
    aligned_data_list = []
    aligned_base_list = []

    has_data = any(d is not None for d in shap_data_list)

    for vals, base, data, names in zip(shap_values_list, shap_base_list, shap_data_list, shap_featnames_list):
        n = vals.shape[0]
        vals_al = np.zeros((n, p_union), dtype=float)
        data_al = np.full((n, p_union), np.nan, dtype=float) if has_data else None

        pos = [name_to_pos[name] for name in names if name in name_to_pos]
        col_indices = [idx for idx, name in enumerate(names) if name in name_to_pos]

        if pos and col_indices:
            vals_al[:, pos] = vals[:, col_indices]
            if data is not None and data_al is not None:
                data_al[:, pos] = data[:, col_indices]

        aligned_vals_list.append(vals_al)
        if data_al is not None:
            aligned_data_list.append(data_al)

        if np.isscalar(base):
            aligned_base_list.append(np.full(n, float(base)))
        else:
            aligned_base_list.append(np.asarray(base, dtype=float).reshape(-1))

    pooled_vals = np.concatenate(aligned_vals_list, axis=0) if aligned_vals_list else np.empty((0, p_union))
    pooled_base = np.concatenate(aligned_base_list, axis=0) if aligned_base_list else np.empty((0,))
    pooled_data = np.concatenate(aligned_data_list, axis=0) if len(aligned_data_list) > 0 else None

    return shap.Explanation(
        values=pooled_vals,
        base_values=pooled_base,
        data=pooled_data,
        feature_names=union_names,
    )


def plot_shap_beeswarm(
    explanation: shap.Explanation,
    title: str = "SHAP Beeswarm Summary Plot",
    save_path: Optional[Union[str, Path]] = None,
    max_display: int = 20,
    show_plot: bool = False,
) -> None:
    """
    Plot and save a publication-quality SHAP beeswarm summary plot for feature importance.
    Shows the distribution of impacts that top features have on the model prediction.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP Explanation object containing values, data, base_values, and feature_names.
    title : str
        Plot title.
    save_path : Optional[Union[str, Path]]
        Destination path. Supports .pdf and automatically saves companion .png.
    max_display : int
        Maximum number of top features to display on the beeswarm plot.
    show_plot : bool
        Whether to display the plot interactively.
    """
    if explanation is None or explanation.values is None or explanation.values.size == 0:
        return

    num_features = explanation.values.shape[1]
    if num_features == 0:
        return

    top_k = min(max_display, num_features)
    mean_abs = np.nanmean(np.abs(explanation.values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_shap = explanation[:, top_idx]

    # Calculate figure height dynamically based on feature count
    fig_height = max(5.0, top_k * 0.35 + 1.8)
    plt.figure(figsize=(10, fig_height))

    try:
        shap.plots.beeswarm(top_shap, max_display=top_k, show=False)
        plt.title(title, fontsize=13, fontweight="bold", pad=15)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=400, bbox_inches="tight")

            # Save companion png or pdf for versatility
            if p.suffix.lower() == ".pdf":
                plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
            elif p.suffix.lower() == ".png":
                plt.savefig(p.with_suffix(".pdf"), dpi=400, bbox_inches="tight")

        if show_plot:
            plt.show()
    except Exception as e:
        warnings.warn(f"SHAP beeswarm plotting failed: {e}. Falling back to bar summary.")
        try:
            plt.clf()
            shap.plots.bar(top_shap, max_display=top_k, show=False)
            plt.title(title, fontsize=13, fontweight="bold", pad=15)
            if save_path:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(p, dpi=400, bbox_inches="tight")
                if p.suffix.lower() == ".pdf":
                    plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
                elif p.suffix.lower() == ".png":
                    plt.savefig(p.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
        except Exception:
            pass
    finally:
        if not show_plot:
            plt.close("all")
        else:
            plt.close()
