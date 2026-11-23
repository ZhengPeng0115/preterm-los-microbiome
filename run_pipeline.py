"""
Top-level pipeline execution script.
Executes and compares:
  (a) Case-control (1:3) study across sources (identifies Best Source)
  (b) Full cohort test on all controls
  (c) Full cohort test on all controls + other sepsis
All evaluations report pooled held-out metrics with 100-run bootstrap 95% CIs
and generate SHAP beeswarm summary plots for overall sepsis and pathogen models in Experiment (a).
"""

import argparse
import io
import sys
import warnings
from pathlib import Path
import pandas as pd

# Guard against running in windowed environments (pythonw.exe) or non-console debuggers where stdout/stderr is None
if sys.stdout is None:
    sys.stdout = sys.__stdout__ or io.StringIO()
if sys.stderr is None:
    sys.stderr = sys.__stderr__ or io.StringIO()

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")

from sepsis_prediction.config import PipelineConfig
from sepsis_prediction.data_loader import SepsisDataLoader
from sepsis_prediction.experiments import (
    run_case_control_study,
    run_full_cohort_study,
)


def format_metric_ci(ci_info: dict) -> str:
    """Format point estimate and 95% CI as string: point [lower - upper]."""
    point = ci_info.get("point", float("nan"))
    lower = ci_info.get("ci_lower", float("nan"))
    upper = ci_info.get("ci_upper", float("nan"))
    return f"{point:.3f} [{lower:.3f} - {upper:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Multi-modal Sepsis Prediction Pipeline")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["a", "b", "c"],
        choices=["a", "b", "c", "all"],
        help="Experiments to run: (a) case-control, (b) full cohort controls, (c) full cohort all",
    )
    parser.add_argument(
        "--pathogens",
        nargs="+",
        default=None,
        help="List of specific pathogens to evaluate. Default: standard 5 pathogens",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run mode with reduced iterations for testing and verification",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full analysis runs even when running inside a debugger",
    )
    parser.add_argument(
        "--no-shap-beeswarm",
        action="store_true",
        help="Disable SHAP beeswarm summary plot generation",
    )
    args = parser.parse_args()

    # Configure pipeline
    cfg = PipelineConfig()

    if args.no_shap_beeswarm:
        cfg.plot_shap_beeswarm = False

    # Auto-detect interactive debugger (e.g. PyCharm pydevd)
    is_debugging = hasattr(sys, "gettrace") and sys.gettrace() is not None
    if is_debugging and not args.quick and not args.full and "--experiments" not in sys.argv:
        print("\n[NOTE] PyCharm Debugger (pydevd) detected without explicit arguments.")
        print("       Enabling fast interactive debug mode (reduced iterations for responsive stepping).")
        print("       To force full 100-run bootstrap inside the debugger, pass --full.\n")
        args.quick = True

    if args.quick:
        cfg.bayes_iter = 3
        cfg.shap_bootstrap_runs = 5
        cfg.n_bootstrap_ci = 10
        cfg.n_splits_outer = 3
        print("\n*** Quick testing mode enabled (reduced CV folds and iterations) ***\n")

    selected_exps = set(args.experiments)
    if "all" in selected_exps:
        selected_exps = {"a", "b", "c"}

    pathogens = args.pathogens or cfg.default_specific_pathogens

    print("=" * 80)
    print("SEPSIS PREDICTION MODULAR PIPELINE")
    print(f"Target pathogens: {pathogens}")
    print(f"Active experiments: {sorted(list(selected_exps))}")
    print(f"SHAP beeswarm visualization: {'Enabled' if cfg.plot_shap_beeswarm else 'Disabled'}")
    print("=" * 80)

    # 1. Ingest and align multi-modal data
    print("\nLoading and aligning clinical and multi-modal microbiome datasets...")
    loader = SepsisDataLoader(config=cfg)
    sources_data = loader.load_all_modalities(pathogens=cfg.all_pathogens)

    ref_df = sources_data["clinical"]
    print(f"Alignment complete: {len(ref_df)} unique patients across {len(ref_df['Match_group'].unique())} matched groups.")
    print(f"Modalities available: {list(sources_data.keys())}")

    best_sources = {}
    exp_a_results = None
    exp_b_results = None
    exp_c_results = None

    # 2. Experiment (a): Case-Control Study & Best Source Selection
    if "a" in selected_exps or "b" in selected_exps or "c" in selected_exps:
        print("\n" + "=" * 80)
        print("EXPERIMENT (a): Case-Control (1:3) Evaluation & Best Source Discovery")
        print("=" * 80)
        best_sources, exp_a_results = run_case_control_study(
            sources_data=sources_data,
            pathogens=pathogens,
            config=cfg,
            run_overall=True,
        )

        print("\n--- Summary of Best Data Sources Identified in Experiment (a) ---")
        for target, src in best_sources.items():
            print(f"  {target:25s} -> Best Source: {src}")

    # 3. Experiment (b): Full Cohort on All Controls (Excluding Other Sepsis)
    if "b" in selected_exps:
        print("\n" + "=" * 80)
        print("EXPERIMENT (b): Full Cohort Evaluation (Target Pathogen vs All Controls)")
        print("=" * 80)
        exp_b_results = run_full_cohort_study(
            sources_data=sources_data,
            best_sources=best_sources,
            pathogens=pathogens,
            include_other_sepsis=False,
            config=cfg,
        )

    # 4. Experiment (c): Full Cohort on All Controls + Other Sepsis
    if "c" in selected_exps:
        print("\n" + "=" * 80)
        print("EXPERIMENT (c): Full Cohort Evaluation (Target Pathogen vs All Controls + Other Sepsis)")
        print("=" * 80)
        exp_c_results = run_full_cohort_study(
            sources_data=sources_data,
            best_sources=best_sources,
            pathogens=pathogens,
            include_other_sepsis=True,
            config=cfg,
        )

    # 5. Output Comprehensive Metrics Table with Pooled 95% CIs
    print("\n" + "=" * 120)
    print("POOLED METRICS TABLE WITH 95% CONFIDENCE INTERVALS (100 Bootstrap Runs)")
    print("=" * 120)

    detailed_rows = []

    # (i) Overall Sepsis Case-Control Results
    if exp_a_results and "overall" in exp_a_results:
        best_overall_src = best_sources.get("overall", "clinical")
        ov_res = exp_a_results["overall"].get(best_overall_src, {})
        ov_ci = ov_res.get("metrics_ci", {})
        detailed_rows.append({
            "Target": "Overall Sepsis (LOS)",
            "Experiment": "(a) Case-Control (1:3)",
            "Data Source": best_overall_src,
            "Threshold": f"{ov_res.get('threshold', 0.5):.3f}",
            "AUC (95% CI)": format_metric_ci(ov_ci.get("auc", {})),
            "Sensitivity (95% CI)": format_metric_ci(ov_ci.get("sensitivity", {})),
            "Specificity (95% CI)": format_metric_ci(ov_ci.get("specificity", {})),
            "PPV (95% CI)": format_metric_ci(ov_ci.get("ppv", {})),
            "NPV (95% CI)": format_metric_ci(ov_ci.get("npv", {})),
        })

    # (ii) Specific Pathogens across Experiments (a), (b), (c)
    for p in pathogens:
        p_clean = p.strip()
        best_src = best_sources.get(p, "clinical")

        # Experiment (a)
        if exp_a_results and p in exp_a_results:
            a_res = exp_a_results[p].get(best_src, {})
            a_ci = a_res.get("metrics_ci", {})
            detailed_rows.append({
                "Target": p_clean,
                "Experiment": "(a) Case-Control (1:3)",
                "Data Source": best_src,
                "Threshold": f"{a_res.get('threshold', 0.5):.3f}",
                "AUC (95% CI)": format_metric_ci(a_ci.get("auc", {})),
                "Sensitivity (95% CI)": format_metric_ci(a_ci.get("sensitivity", {})),
                "Specificity (95% CI)": format_metric_ci(a_ci.get("specificity", {})),
                "PPV (95% CI)": format_metric_ci(a_ci.get("ppv", {})),
                "NPV (95% CI)": format_metric_ci(a_ci.get("npv", {})),
            })

        # Experiment (b)
        if exp_b_results and p in exp_b_results:
            b_res = exp_b_results[p]
            b_ci = b_res.get("metrics_ci", {})
            detailed_rows.append({
                "Target": p_clean,
                "Experiment": "(b) Full Cohort (All Controls)",
                "Data Source": best_src,
                "Threshold": f"{b_res.get('best_threshold', 0.5):.3f}",
                "AUC (95% CI)": format_metric_ci(b_ci.get("auc", {})),
                "Sensitivity (95% CI)": format_metric_ci(b_ci.get("sensitivity", {})),
                "Specificity (95% CI)": format_metric_ci(b_ci.get("specificity", {})),
                "PPV (95% CI)": format_metric_ci(b_ci.get("ppv", {})),
                "NPV (95% CI)": format_metric_ci(b_ci.get("npv", {})),
            })

        # Experiment (c)
        if exp_c_results and p in exp_c_results:
            c_res = exp_c_results[p]
            c_ci = c_res.get("metrics_ci", {})
            detailed_rows.append({
                "Target": p_clean,
                "Experiment": "(c) Full Cohort (All Controls + Other Sepsis)",
                "Data Source": best_src,
                "Threshold": f"{c_res.get('best_threshold', 0.5):.3f}",
                "AUC (95% CI)": format_metric_ci(c_ci.get("auc", {})),
                "Sensitivity (95% CI)": format_metric_ci(c_ci.get("sensitivity", {})),
                "Specificity (95% CI)": format_metric_ci(c_ci.get("specificity", {})),
                "PPV (95% CI)": format_metric_ci(c_ci.get("ppv", {})),
                "NPV (95% CI)": format_metric_ci(c_ci.get("npv", {})),
            })

    detailed_df = pd.DataFrame(detailed_rows)
    print(detailed_df.to_string(index=False))

    # Save detailed metrics table to figures directory
    detailed_table_path = Path("figures") / "pooled_metrics_detailed_table.csv"
    detailed_table_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(detailed_table_path, index=False)
    print(f"\nSaved detailed metrics table to: {detailed_table_path}")

    if cfg.plot_shap_beeswarm:
        print("\n" + "=" * 80)
        print("SHAP BEESWARM SUMMARY PLOTS")
        print("=" * 80)
        print("Feature importance beeswarm plots saved with companion PNG and PDF copies in 'figures/':")
        if "a" in selected_exps:
            print("  - Exp (a) Case-Control: figures/exp_a_<target>_<source>_shap_beeswarm.pdf (.png)")


if __name__ == "__main__":
    main()
