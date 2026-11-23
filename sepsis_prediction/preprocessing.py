"""
Preprocessing and feature engineering module.
Contains split-aware clinical preprocessing to prevent data leakage across CV folds,
as well as longitudinal microbiome feature extraction with zero-variance protection.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from skbio.stats.composition import closure, clr, multi_replace


class ClinicalSplitPreprocessor(BaseEstimator, TransformerMixin):
    """
    Split-aware preprocessor for clinical tabular data.
    Ensures that imputation statistics (continuous median, nominal simple mode)
    and one-hot encodings are computed strictly on training splits to prevent data leakage.
    """

    def __init__(
        self,
        nominal_features: List[str],
        one_hot_features: List[str],
        scale_features: Optional[List[str]] = None,
        one_hot_categories: Optional[Dict[str, Sequence]] = None,
        random_state: int = 33,
    ):
        self.nominal_features = list(nominal_features)
        self.one_hot_features = list(one_hot_features)
        self.scale_features = list(scale_features) if scale_features else []
        self.one_hot_categories = dict(one_hot_categories) if one_hot_categories else {}
        self.random_state = random_state

        self.num_imputer_: Optional[SimpleImputer] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.nominal_modes_: Dict[str, any] = {}
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers and encoders on training data only."""
        df = X.copy()
        df = df.replace({-99: np.nan, -99.0: np.nan})

        # Infer scale features if not provided
        meta_cols = {"Pt_ID", "Match_group", "general sepsis", "source sepsis", "pathogen", "LOS"}
        if not self.scale_features:
            self.scale_features = [
                col for col in df.columns
                if col not in self.nominal_features and col not in meta_cols and not col.startswith("label")
            ]

        # 1. Fit numerical imputer (median) on scale features
        if self.scale_features:
            self.num_imputer_ = SimpleImputer(strategy="median")
            self.num_imputer_.fit(df[self.scale_features])

        # 2. Fit simple mode for nominal features
        self.nominal_modes_ = {}
        for col in self.nominal_features:
            if col in df.columns:
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    mode_series = valid_vals.mode(dropna=True)
                    self.nominal_modes_[col] = mode_series.iloc[0] if len(mode_series) > 0 else valid_vals.iloc[0]
                else:
                    self.nominal_modes_[col] = 0

        # 3. Fit OneHotEncoder on training nominal/one-hot features
        if self.one_hot_features:
            # Impute nominals before fitting encoder
            df_nom = df[self.one_hot_features].copy()
            for col in self.one_hot_features:
                if col in self.nominal_modes_:
                    df_nom[col] = df_nom[col].fillna(self.nominal_modes_[col])

            if self.one_hot_categories and all(col in self.one_hot_categories for col in self.one_hot_features):
                cat_list = [sorted(list(self.one_hot_categories[col])) for col in self.one_hot_features]
                self.encoder_ = OneHotEncoder(categories=cat_list, drop="first", sparse_output=False, handle_unknown="ignore")
            else:
                self.encoder_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

            self.encoder_.fit(df_nom)

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform dataset using parameters learned strictly during fit."""
        df = X.copy()
        df = df.replace({-99: np.nan, -99.0: np.nan})

        transformed_parts = []
        feature_names = []

        # 1. Transform scale features using fitted median
        if self.scale_features and self.num_imputer_ is not None:
            num_data = self.num_imputer_.transform(df[self.scale_features])
            transformed_parts.append(num_data)
            feature_names.extend(self.scale_features)

        # 2. Impute nominal features using fitted training modes
        imputed_nominal_dfs = {}
        for col in self.nominal_features:
            if col in df.columns:
                series = df[col].copy()
                mode_val = self.nominal_modes_.get(col, 0)
                series = series.fillna(mode_val)
                imputed_nominal_dfs[col] = series

        # 3. Transform one-hot features with warning suppression for unknown categories
        if self.one_hot_features and self.encoder_ is not None:
            df_oh = pd.DataFrame(
                {col: imputed_nominal_dfs.get(col, df[col]) for col in self.one_hot_features},
                index=df.index
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Found unknown categories.*",
                    category=UserWarning,
                )
                oh_data = self.encoder_.transform(df_oh)
            oh_names = self.encoder_.get_feature_names_out(self.one_hot_features).tolist()
            transformed_parts.append(oh_data)
            feature_names.extend(oh_names)

        # 4. Include remaining nominal features that were not one-hot encoded
        remaining_nominals = [col for col in self.nominal_features if col not in self.one_hot_features and col in df.columns]
        if remaining_nominals:
            nom_array = np.column_stack([imputed_nominal_dfs[col].to_numpy(dtype=float) for col in remaining_nominals])
            transformed_parts.append(nom_array)
            feature_names.extend(remaining_nominals)

        if not transformed_parts:
            return np.empty((len(df), 0)), []

        X_transformed = np.column_stack(transformed_parts)
        self.feature_names_out_ = feature_names
        return X_transformed, feature_names


def filter_low_prevalence_taxa(
    df: pd.DataFrame,
    microbiome_cols: List[str],
    prevalence_threshold: float = 0.05
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter out taxa with sample prevalence below threshold."""
    microbiome_data = df[microbiome_cols]
    prevalence = (microbiome_data > 0).mean(axis=0)
    keep_taxa = prevalence[prevalence >= prevalence_threshold].index.tolist()
    other_cols = [col for col in df.columns if col not in microbiome_cols]
    df_filtered = df[other_cols + keep_taxa].copy()
    return df_filtered, keep_taxa


def alpha_core(p_nz: pd.Series) -> Dict[str, float]:
    """Calculate sample-level alpha diversity metrics: Shannon H and Dominance."""
    p_vals = p_nz.to_numpy(dtype=float)
    p_vals = p_vals[p_vals > 0]
    if len(p_vals) == 0:
        return {"H": 0.0, "dominance": 0.0}
    H = float(-(p_vals * np.log(p_vals)).sum())
    dominance = float(p_vals.max())
    return {"H": H, "dominance": dominance}


def calculate_individual_microbiome_features(
    patient_data: pd.DataFrame,
    microbiome_cols: List[str],
    time_col: str = "Day_of_sampling"
) -> Dict[str, float]:
    """
    Extract longitudinal features (mean, max, last, slope) for individual taxa.
    Includes zero-variance checks on both time and abundance to prevent ValueError in linregress.
    """
    features = {}
    patient_data_sorted = patient_data.sort_values(time_col, ascending=True)
    microbiome_data = patient_data_sorted[microbiome_cols]

    mean_values = microbiome_data.mean()
    max_values = microbiome_data.max()
    last_sample = patient_data_sorted.iloc[-1]

    for col in microbiome_cols:
        features[f"{col}_mean"] = float(mean_values[col])
        features[f"{col}_max"] = float(max_values[col])
        features[f"{col}_last"] = float(last_sample[col])

    # Longitudinal slope with robust zero-variance protection
    time_values = patient_data_sorted[time_col].to_numpy(dtype=float)
    has_valid_time = (
        len(patient_data_sorted) >= 2
        and len(np.unique(time_values)) > 1
        and float(np.std(time_values)) > 1e-6
    )

    for col in microbiome_cols:
        abundance_values = patient_data_sorted[col].to_numpy(dtype=float)
        if not has_valid_time or float(np.std(abundance_values)) < 1e-6:
            features[f"{col}_slope"] = 0.0
        else:
            try:
                res = linregress(time_values, abundance_values)
                slope = float(res.slope) if np.isfinite(res.slope) else 0.0
                features[f"{col}_slope"] = slope
            except ValueError:
                features[f"{col}_slope"] = 0.0

    return features


def calculate_diversity_features(
    patient_data: pd.DataFrame,
    microbiome_cols: List[str],
    time_col: str = "Day_of_sampling"
) -> Dict[str, float]:
    """
    Extract longitudinal diversity features (Shannon and Dominance).
    Guards against linregress zero variance errors.
    """
    features = {}
    patient_data_sorted = patient_data.sort_values(time_col, ascending=True)

    H, D = [], []
    for _, sample in patient_data_sorted[microbiome_cols].iterrows():
        m = alpha_core(sample)
        H.append(m["H"])
        D.append(m["dominance"])

    H = np.array(H, dtype=float)
    D = np.array(D, dtype=float)

    features["Shannon_mean"] = float(np.mean(H))
    features["Dominance_mean"] = float(np.mean(D))
    features["Shannon_max"] = float(np.max(H))
    features["Dominance_max"] = float(np.max(D))
    features["Shannon_last"] = float(H[-1])
    features["Dominance_last"] = float(D[-1])

    time_values = patient_data_sorted[time_col].to_numpy(dtype=float)
    has_valid_time = (
        len(patient_data_sorted) >= 2
        and len(np.unique(time_values)) > 1
        and float(np.std(time_values)) > 1e-6
    )

    if not has_valid_time:
        features["Shannon_slope"] = 0.0
        features["Dominance_slope"] = 0.0
    else:
        # Shannon slope
        if float(np.std(H)) < 1e-6:
            features["Shannon_slope"] = 0.0
        else:
            try:
                res_h = linregress(time_values, H)
                features["Shannon_slope"] = float(res_h.slope) if np.isfinite(res_h.slope) else 0.0
            except ValueError:
                features["Shannon_slope"] = 0.0

        # Dominance slope
        if float(np.std(D)) < 1e-6:
            features["Dominance_slope"] = 0.0
        else:
            try:
                res_d = linregress(time_values, D)
                features["Dominance_slope"] = float(res_d.slope) if np.isfinite(res_d.slope) else 0.0
            except ValueError:
                features["Dominance_slope"] = 0.0

    return features


def aggregate_patient_features(patient_data: pd.DataFrame, microbiome_cols: List[str]) -> Dict[str, float]:
    """Aggregate all sample-level features into patient-level summary."""
    all_features = {"Pt_ID": patient_data["Pt_ID"].iloc[0]}

    col_individual = [c for c in microbiome_cols if ("prop" not in c) and ("Unknown" not in c)]
    microbiome_features = calculate_individual_microbiome_features(patient_data, col_individual)
    all_features.update(microbiome_features)

    col_diversity = [c for c in microbiome_cols if "prop" in c]
    diversity_features = calculate_diversity_features(patient_data, col_diversity)
    all_features.update(diversity_features)

    return all_features


def process_microbiome(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate longitudinal microbiome samples to patient-level features."""
    df = df.fillna(0)
    metadata_cols = ["Pt_ID", "Sample_ID", "Day_of_sampling", "Day_before_sepsis", "Match_group"]
    microbiome_cols = [col for col in df.columns if col not in metadata_cols]

    patient_features = []
    for patient_id in df["Pt_ID"].unique():
        patient_data = df[df["Pt_ID"] == patient_id]
        features = aggregate_patient_features(patient_data, microbiome_cols)
        patient_features.append(features)

    return pd.DataFrame(patient_features)
