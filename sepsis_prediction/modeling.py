"""
Modeling and hyperparameter optimization module.
Handles XGBoost classification with adaptive inner cross-validation,
calibrated scale_pos_weight, dynamic upper bounds on min_child_weight,
and small-sample stability protections.
"""

import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier

from .config import PipelineConfig


def get_adaptive_inner_cv(
    y_train: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits_inner: int = 3,
    random_state: int = 33,
):
    """
    Construct inner cross-validation strategy adapted to positive case count.
    Prevents degenerate folds with 0 or 1 positive case.
    """
    n_positives = int((y_train == 1).sum())
    # Ensure n_splits does not exceed positive case count
    splits = max(2, min(n_splits_inner, n_positives))

    # Check if groups are provided and sufficient unique groups exist per class
    if groups is not None and len(np.unique(groups)) >= splits * 2:
        try:
            cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)
            # Test split viability
            next(cv.split(np.zeros((len(y_train), 1)), y_train, groups))
            return cv
        except Exception:
            pass

    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def train_xgboost_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Optional[np.ndarray] = None,
    config: Optional[PipelineConfig] = None,
    random_state: int = 33,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost classifier with Bayesian hyperparameter optimization.
    Uses fixed, analytically determined scale_pos_weight to prevent probability distortions
    across outer cross-validation folds.
    Applies dynamic upper bounds on min_child_weight to prevent empty trees on small cohorts.
    Safely adjusts parallel jobs to 1 when PyCharm debugger is active to avoid multiprocessing crashes.
    """
    cfg = config or PipelineConfig()
    n_samples, n_features = X_train.shape
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    # When running in a debugger (e.g. PyCharm pydevd), use single process to prevent BrokenProcessPool crashes
    is_debugging = hasattr(sys, "gettrace") and sys.gettrace() is not None
    parallel_jobs = 1 if is_debugging else -1

    # Analytical class weighting: (neg / pos) maintains consistent calibration across folds
    scale_pos_weight = float(n_neg / max(1, n_pos))

    # Dynamic upper bound on min_child_weight based on positive events:
    # If min_child_weight exceeds the positive class hessian,
    # the tree is mathematically prohibited from splitting, creating empty trees (0:leaf=0).
    max_mcw = max(1, min(10, n_pos // 2))

    # If positive events are extremely scarce (<= 3), skip noisy Bayesian search and use robust defaults
    if n_pos <= 3 or n_samples < 15:
        mcw = max(1, min(2, n_pos // 2)) if n_pos > 1 else 1
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            learning_rate=0.03,
            n_estimators=100,
            max_depth=3,
            min_child_weight=mcw,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=2.0,
            gamma=0.0,
            n_jobs=parallel_jobs,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model, {"learning_rate": 0.03, "n_estimators": 100, "max_depth": 3, "min_child_weight": mcw}

    # Adaptive inner CV
    inner_cv = get_adaptive_inner_cv(
        y_train,
        groups=groups,
        n_splits_inner=cfg.n_splits_inner,
        random_state=random_state,
    )

    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        max_delta_step=1.0,
        n_jobs=parallel_jobs,
        random_state=random_state,
    )

    # Search spaces with dynamic min_child_weight upper bound and small-sample gamma protection
    search_spaces = dict(cfg.search_spaces)
    if max_mcw < 10:
        search_spaces["min_child_weight"] = Integer(1, max_mcw)
    if n_pos <= 7:
        search_spaces["gamma"] = Real(0.0, 2.0)

    # Use average_precision or roc_auc depending on class imbalance
    scoring = "roc_auc" if n_pos >= 5 else "balanced_accuracy"

    opt = BayesSearchCV(
        base_xgb,
        search_spaces,
        n_iter=cfg.bayes_iter,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=parallel_jobs,
        random_state=random_state,
        verbose=0,
        refit=True,
    )

    try:
        if isinstance(inner_cv, StratifiedGroupKFold) and groups is not None:
            opt.fit(X_train, y_train, groups=groups)
        else:
            opt.fit(X_train, y_train)
        best_model = opt.best_estimator_
        best_params = opt.best_params_
    except Exception:
        # Fallback to base model if Bayesian search fails
        base_xgb.fit(X_train, y_train)
        best_model = base_xgb
        best_params = {}

    return best_model, best_params


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
) -> np.ndarray:
    """Compute predicted positive probabilities for test samples."""
    return model.predict_proba(X_test)[:, 1]
