"""
Configuration module for the Sepsis Prediction Pipeline.
Defines default paths, hyperparameter search spaces, pathogen groupings,
and cross-validation settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from skopt.space import Real, Integer


@dataclass
class PipelineConfig:
    """Configuration settings for sepsis prediction modeling and evaluation."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    metadata_file: str = "Metadata_sepsis_prediction_23102025.xlsx"
    figures_dir: str = "figures"

    # Microbe data files relative to parent folder
    microbiome_files: Dict[str, str] = field(default_factory=lambda: {
        "microbiome_peak": "4. peaks_AA_sepsis prediction.csv",
        "microbiome_species": "3. species_AA_sepsis prediction.csv",
        "microbiome_genus": "2. genus_AA_sepsis prediction.csv",
        "microbiome_phylum": "phylum_AA_sepsis prediction.csv",
    })

    # Pathogen specifications
    all_pathogens: List[str] = field(default_factory=lambda: [
        "Klebsiella spp. ",
        "Enterobacter cloacae",
        "Escherichia coli",
        "Serratia marcecens",
        "Staphylococcus aureus",
        "Bacillus species",
        "Bacillus cereus",
        "Enterococcus faecalis",
        "Group B streptococcus",
        "Pantoea septica",
        "Proteus mirabilis",
        "Pseudomonas spp.",
    ])

    default_specific_pathogens: List[str] = field(default_factory=lambda: [
        "Serratia marcecens",
        "Escherichia coli",
        "Klebsiella spp. ",
        "Enterobacter cloacae",
        "Staphylococcus aureus",
    ])

    # Clinical feature definitions
    nominal_features: List[str] = field(default_factory=lambda: [
        "Hospital_site",
        "Biological_sex",
        "Birth_mode",
        "Singleton",
        "Corticosteroids_mother",
        "Surfactant_composit_72hours",
        "AB_exposure_before_onset_sepsis",
    ])

    one_hot_features: List[str] = field(default_factory=lambda: [
        "Hospital_site",
    ])

    one_hot_categories: Dict[str, List] = field(default_factory=lambda: {
        "Hospital_site": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })

    clinical_cols: List[str] = field(default_factory=lambda: [
        "Hospital_site",
        "Biological_sex",
        "Birth_mode",
        "Singleton",
        "GA_total",
        "Postnatal_age_sepsis",
        "Birthweight",
        "Apgar_score_1min",
        "Apgar_score_5min",
        "Corticosteroids_mother",
        "Surfactant_composit_72hours",
        "AB_exposure_before_onset_sepsis",
        "AB_IV_days_before_onset_sepsis",
        "AB_IV_days_before_onset_sepsis_ratio",
        "Full_enteral_feeding_before_onset_sepsis",
    ])

    # Cross-validation & sampling settings
    random_seed: int = 0
    n_splits_outer: int = 5
    n_splits_inner: int = 3
    bayes_iter: int = 20

    # Feature selection
    shap_k: int = 30
    shap_bootstrap_runs: int = 30
    shap_sample_fraction: float = 0.8
    adaptive_feature_selection: bool = False  # Consistent 30 features across all targets (or p if p < 30)

    # Uncertainty & Bootstrap evaluation
    n_bootstrap_ci: int = 100
    ci_level: float = 0.95

    # SHAP Beeswarm summary visualization
    plot_shap_beeswarm: bool = True
    shap_beeswarm_max_display: int = 10

    # Bayesian search space for XGBoost
    search_spaces: Dict = field(default_factory=lambda: {
        "learning_rate": Real(1e-3, 0.1, prior="log-uniform"),
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 6),
        "min_child_weight": Integer(1, 10),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 0.9),
        "gamma": Real(0, 5),
        "reg_lambda": Real(1e-2, 10, prior="log-uniform"),
    })

    def get_metadata_path(self) -> Path:
        """Resolve metadata file path."""
        candidate = self.project_root / self.metadata_file
        if candidate.exists():
            return candidate
        parent_candidate = self.project_root.parent / self.metadata_file
        if parent_candidate.exists():
            return parent_candidate
        return candidate

    def get_microbiome_path(self, source_name: str) -> Path:
        """Resolve path to a microbiome raw file."""
        filename = self.microbiome_files[source_name]
        candidate = self.project_root / filename
        if candidate.exists():
            return candidate
        parent_candidate = self.project_root.parent / filename
        if parent_candidate.exists():
            return parent_candidate
        return candidate
