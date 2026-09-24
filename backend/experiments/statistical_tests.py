"""Run Kolmogorov-Smirnov and Chi-Square statistical tests for distribution shifts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import load_dataset, FEATURE_NAMES


def get_effect_size_label(val: float, is_binary: bool) -> str:
    if is_binary:
        # Cramér's V / Phi effect size thresholds for 1 DOF (2x2 table)
        # Small: 0.1, Medium: 0.3, Large: 0.5
        if val >= 0.5:
            return "Very High"
        elif val >= 0.3:
            return "High"
        elif val >= 0.1:
            return "Moderate"
        else:
            return "Negligible"
    else:
        # KS statistic acts as a maximum vertical difference (0 to 1)
        if val >= 0.5:
            return "Very High"
        elif val >= 0.25:
            return "High"
        elif val >= 0.1:
            return "Moderate"
        else:
            return "Negligible"


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical tests for distribution shifts.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument(
        "--external-dataset",
        type=Path,
        default=experiments_dir / "external_wangchuk.csv",
        help="Path to external Wangchuk dataset CSV."
    )
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample of {args.sample_size:,} rows from PhiUSIIL...")
    phi_df = load_dataset(args.dataset, args.sample_size, args.seed)
    
    print(f"Loading sample of {args.sample_size:,} rows from Wangchuk...")
    wang_df = load_dataset(args.external_dataset, args.sample_size, args.seed)

    results = []

    print("\nRunning statistical tests...")
    for feature in FEATURE_NAMES:
        is_binary = feature in ["is_https", "has_ip"]
        phi_data = phi_df[feature].to_numpy()
        wang_data = wang_df[feature].to_numpy()

        if is_binary:
            # Contingency table (2x2)
            #           PhiUSIIL   Wangchuk
            # Value 0:   count0     count0
            # Value 1:   count1     count1
            phi_1_cnt = int(np.sum(phi_data == 1))
            phi_0_cnt = len(phi_data) - phi_1_cnt
            wang_1_cnt = int(np.sum(wang_data == 1))
            wang_0_cnt = len(wang_data) - wang_1_cnt

            table = [[phi_0_cnt, wang_0_cnt], [phi_1_cnt, wang_1_cnt]]
            chi2, p_val, _, _ = chi2_contingency(table)
            
            # Cramér's V for 2x2 contingency table (dof = 1)
            total_n = len(phi_data) + len(wang_data)
            v = np.sqrt(chi2 / total_n)
            
            results.append({
                "feature": feature,
                "test": "Chi-Square",
                "statistic": float(chi2),
                "p_value": float(p_val),
                "effect_size": float(v),
                "effect_label": get_effect_size_label(v, is_binary=True)
            })
            print(f"  {feature:<20} | Chi-Square | p-val: {p_val:.3e} | Cramér's V: {v:.4f} ({results[-1]['effect_label']})")
        else:
            # KS test
            stat, p_val = ks_2samp(phi_data, wang_data)
            results.append({
                "feature": feature,
                "test": "Kolmogorov-Smirnov",
                "statistic": float(stat),
                "p_value": float(p_val),
                "effect_size": float(stat),
                "effect_label": get_effect_size_label(stat, is_binary=False)
            })
            print(f"  {feature:<20} | KS-Test    | p-val: {p_val:.3e} | KS Stat:   {stat:.4f} ({results[-1]['effect_label']})")

    output_path = args.output_dir / "statistical_tests.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nStatistical test results saved to {output_path}")

    # Generate markdown table
    md_lines = [
        "| Feature | Test Method | Statistic | p-value | Effect Size | Distribution Shift |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    for res in results:
        stat_name = "χ²" if res["test"] == "Chi-Square" else "KS"
        p_str = f"< 0.001" if res["p_value"] < 0.001 else f"{res['p_value']:.4f}"
        md_lines.append(
            f"| {res['feature'].replace('_', ' ').title()} | {res['test']} | {stat_name} = {res['statistic']:.2f} | {p_str} | {res['effect_size']:.4f} | {res['effect_label']} |"
        )
    
    md_path = args.output_dir / "statistical_tests_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
