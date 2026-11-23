"""
Data loading and cohort assembly module.
Handles multi-modal data ingestion (clinical, peak, species, genus, phylum)
and cohort alignment across modalities.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from skbio.stats.composition import closure, clr, multi_replace

from .config import PipelineConfig
from .preprocessing import filter_low_prevalence_taxa, process_microbiome


class SepsisDataLoader:
    """Manages loading, aligning, and cohort filtering across clinical and microbiome sources."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def load_clinical(
        self,
        pathogens: Union[str, List[str]],
        label_name: str = "LOS",
    ) -> pd.DataFrame:
        """
        Load raw clinical data and Maintains columns for split-aware preprocessing.
        """
        meta_path = self.config.get_metadata_path()
        cols = list(self.config.clinical_cols) + ["Pt_ID", "Match_group", "Variable_3"]

        df = pd.read_excel(meta_path, sheet_name="Metadata_patient-level", usecols=cols)

        if isinstance(pathogens, str):
            pathogens = [pathogens]

        if len(pathogens) > 1:
            group_keep = df.loc[df["Variable_3"].isin(pathogens), "Match_group"].unique()
            df = df.loc[df["Match_group"].isin(group_keep)].rename(columns={"Variable_3": label_name}).reset_index(drop=True)
            df["pathogen"] = df[label_name]
            df[label_name] = df[label_name].apply(lambda x: 1 if x in pathogens else 0)
        else:
            p_target = pathogens[0]
            group_keep = df.loc[df["Variable_3"] == p_target, "Match_group"].unique()
            df = df.loc[df["Match_group"].isin(group_keep)].rename(columns={"Variable_3": label_name}).reset_index(drop=True)
            df["pathogen"] = df[label_name]
            df[label_name] = (df[label_name] == p_target).astype(int)

        return df

    def load_microbiome(
        self,
        source: str,
        pathogens: Union[str, List[str]],
        label_name: str = "LOS",
    ) -> pd.DataFrame:
        """
        Load and process raw microbiome feature tables.
        Applies prevalence filtering, CLR transform, and patient-level longitudinal aggregation.
        """
        if source not in self.config.microbiome_files:
            raise ValueError(f"Unknown microbiome source: {source}. Valid sources: {list(self.config.microbiome_files.keys())}")

        file_path = self.config.get_microbiome_path(source)
        df_raw = pd.read_csv(file_path, header=0).rename(columns={"Patient_ID": "Sample_ID"})

        meta_path = self.config.get_metadata_path()
        df_sample_info = pd.read_excel(
            meta_path,
            sheet_name="Metadata_sample-level",
            usecols=["Sample_ID", "Pt_ID", "Timeperiod", "Day_of_sampling", "Day_before_sepsis", "Match_group", "Variable_3"],
        )
        df_patient_info = pd.read_excel(
            meta_path,
            sheet_name="Metadata_patient-level",
            usecols=["Pt_ID", "Match_group", "Variable_3"],
        )

        if isinstance(pathogens, str):
            pathogens = [pathogens]

        if len(pathogens) > 1:
            group_keep_sample = df_sample_info.loc[df_sample_info["Variable_3"].isin(pathogens), "Match_group"].unique()
            df_sample_info = df_sample_info.loc[df_sample_info["Match_group"].isin(group_keep_sample)].rename(
                columns={"Variable_3": label_name}
            ).reset_index(drop=True)
            df_sample_info[label_name] = df_sample_info[label_name].apply(lambda x: 1 if x in pathogens else 0)

            group_keep_patient = df_patient_info.loc[df_patient_info["Variable_3"].isin(pathogens), "Match_group"].unique()
            df_patient_info = df_patient_info.loc[df_patient_info["Match_group"].isin(group_keep_patient)].rename(
                columns={"Variable_3": label_name}
            ).reset_index(drop=True)
            df_patient_info["pathogen"] = df_patient_info[label_name]
            df_patient_info[label_name] = df_patient_info[label_name].apply(lambda x: 1 if x in pathogens else 0)
        else:
            p_target = pathogens[0]
            group_keep_sample = df_sample_info.loc[df_sample_info["Variable_3"] == p_target, "Match_group"].unique()
            df_sample_info = df_sample_info.loc[df_sample_info["Match_group"].isin(group_keep_sample)].rename(
                columns={"Variable_3": label_name}
            ).reset_index(drop=True)
            df_sample_info[label_name] = (df_sample_info[label_name] == p_target).astype(int)

            group_keep_patient = df_patient_info.loc[df_patient_info["Variable_3"] == p_target, "Match_group"].unique()
            df_patient_info = df_patient_info.loc[df_patient_info["Match_group"].isin(group_keep_patient)].rename(
                columns={"Variable_3": label_name}
            ).reset_index(drop=True)
            df_patient_info["pathogen"] = df_patient_info[label_name]
            df_patient_info[label_name] = (df_patient_info[label_name] == p_target).astype(int)

        df_raw = df_raw.loc[df_raw["Sample_ID"].isin(df_sample_info["Sample_ID"])]
        df_raw, _ = filter_low_prevalence_taxa(df_raw, df_raw.columns[1:], prevalence_threshold=0.05)
        df_raw = df_raw.loc[:, df_raw.nunique() > 1]
        df_raw = df_raw.loc[~(df_raw.iloc[:, 1:] == 0).all(axis=1)]

        P = closure(df_raw.iloc[:, 1:])
        P_nz = pd.DataFrame(multi_replace(P), index=df_raw.index, columns=df_raw.columns[1:])
        df_RA = pd.concat([df_raw[["Sample_ID"]], pd.DataFrame(P_nz.values, index=P_nz.index, columns=P_nz.columns)], axis=1)
        df_RA_CLR = pd.concat([df_raw[["Sample_ID"]], pd.DataFrame(clr(P_nz.values), index=P_nz.index, columns=P_nz.columns)], axis=1)

        df_merged = pd.merge(df_RA_CLR, df_RA, on="Sample_ID", how="outer", suffixes=("", "_prop"))
        df_sample = pd.merge(
            df_merged,
            df_sample_info[["Sample_ID", "Pt_ID", "Day_of_sampling", "Day_before_sepsis", "Match_group"]],
            on="Sample_ID",
            how="inner",
        )
        df_patient_features = process_microbiome(df_sample)
        df_final = pd.merge(df_patient_features, df_patient_info, on="Pt_ID", how="inner")
        return df_final

    def load_all_modalities(
        self,
        pathogens: Optional[List[str]] = None,
        label_name: str = "LOS",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load clinical and all 4 microbiome modalities, and align across shared patients.
        """
        if pathogens is None:
            pathogens = self.config.all_pathogens

        df_clinical = self.load_clinical(pathogens, label_name=label_name)
        microbiome_dfs = {}
        for source_name in self.config.microbiome_files.keys():
            microbiome_dfs[source_name] = self.load_microbiome(source_name, pathogens, label_name=label_name)

        # Identify shared patients across all 5 modalities
        shared_pt_ids = set(df_clinical["Pt_ID"])
        for m_df in microbiome_dfs.values():
            shared_pt_ids = shared_pt_ids & set(m_df["Pt_ID"])

        # Filter all modalities to shared patients
        df_clinical = df_clinical[df_clinical["Pt_ID"].isin(shared_pt_ids)].copy()
        for s in microbiome_dfs:
            microbiome_dfs[s] = microbiome_dfs[s][microbiome_dfs[s]["Pt_ID"].isin(shared_pt_ids)].copy()

        # Sort each source by Pt_ID to guarantee aligned row ordering
        df_clinical = df_clinical.sort_values("Pt_ID").reset_index(drop=True)
        aligned_sources = {"clinical": df_clinical}
        for s, m_df in microbiome_dfs.items():
            aligned_sources[s] = m_df.sort_values("Pt_ID").reset_index(drop=True)

        return aligned_sources
