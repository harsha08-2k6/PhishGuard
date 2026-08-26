"""Compare feature distributions between PhiUSIIL and Wangchuk datasets."""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
# pyrefly: ignore [missing-import]
from joblib import Parallel, delayed

# Set up path to import extract_features
backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))
from extract_features import FEATURE_ORDER, extract_features

def process_dataset(path: Path, name: str) -> pd.DataFrame:
    print(f"Loading {name} from {path}...")
    # Load URL and label
    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["URL", "label"]
    if not set(required).issubset(available_cols):
        raise ValueError(f"Dataset {name} must contain URL and label.")
    
    df = pd.read_csv(path, usecols=required, low_memory=False)
    df["URL"] = df["URL"].astype(str)
    df = df.dropna(subset=["URL", "label"])
    
    print(f"Extracting features for {name} ({len(df):,} rows) in parallel...")
    urls = df["URL"].tolist()
    features_list = Parallel(n_jobs=-1, batch_size=1000)(
        delayed(extract_features)(url) for url in urls
    )
    
    features_df = pd.DataFrame(features_list)
    combined = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    print(f"Finished feature extraction for {name}.\n")
    return combined

def calculate_stats(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    # Structure: stats[feature][label_val][metric]
    stats = {}
    
    for feature in FEATURE_ORDER:
        stats[feature] = {}
        for label in [0, 1]:
            subset = df[df["label"] == label][feature]
            stats[feature][label] = {}
            if len(subset) == 0:
                continue
            
            # Determine if binary
            is_binary = feature in ["is_https", "has_ip"]
            
            if is_binary:
                # percentage of 1
                pct = float(subset.mean() * 100)
                stats[feature][label]["pct_1"] = pct
            else:
                stats[feature][label]["mean"] = float(subset.mean())
                stats[feature][label]["std"] = float(subset.std())
                stats[feature][label]["min"] = float(subset.min())
                stats[feature][label]["p25"] = float(subset.quantile(0.25))
                stats[feature][label]["median"] = float(subset.median())
                stats[feature][label]["p75"] = float(subset.quantile(0.75))
                stats[feature][label]["p90"] = float(subset.quantile(0.90))
                stats[feature][label]["p95"] = float(subset.quantile(0.95))
                stats[feature][label]["max"] = float(subset.max())
                
    return stats

def format_compact_cell(stats_dict: dict[str, float], is_binary: bool) -> str:
    if not stats_dict:
        return "N/A"
    if is_binary:
        return f"{stats_dict['pct_1']:.2f}%"
    else:
        mean_val = stats_dict["mean"]
        std_val = stats_dict["std"]
        med_val = stats_dict["median"]
        p25 = stats_dict["p25"]
        p75 = stats_dict["p75"]
        p95 = stats_dict["p95"]
        return (
            f"**Mean**: {mean_val:.3f} ± {std_val:.3f}<br>"
            f"**Med**: {med_val:.1f} ({p25:.1f}, {p75:.1f}, {p95:.1f})"
        )

def main() -> None:
    phi_path = Path("C:/Users/91965/Downloads/PhiUSIIL_Phishing_URL_Dataset.csv")
    wang_path = backend_dir / "experiments" / "external_wangchuk.csv"
    
    if not phi_path.exists():
        print(f"Error: PhiUSIIL dataset not found at {phi_path}")
        sys.exit(1)
    if not wang_path.exists():
        print(f"Error: Wangchuk dataset not found at {wang_path}")
        sys.exit(1)
        
    phi_df = process_dataset(phi_path, "PhiUSIIL")
    wang_df = process_dataset(wang_path, "Wangchuk")
    
    # Calculate stats
    phi_stats = calculate_stats(phi_df)
    wang_stats = calculate_stats(wang_df)
    
    # Let's write out a report.
    output_dir = backend_dir / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "feature_comparison_report.md"
    
    report_lines = []
    report_lines.append("# Feature Distribution Analysis Report")
    report_lines.append(f"\nAnalyzed datasets:")
    report_lines.append(f"- **PhiUSIIL**: {len(phi_df):,} URLs")
    report_lines.append(f"- **Wangchuk**: {len(wang_df):,} URLs")
    report_lines.append("\n## Table 1: Compact Comparison Table")
    report_lines.append("\nValues formatted as `Mean ± Std` and `Median (25th, 75th, 95th Percentile)`. Binary features show `Percentage = 1`.")
    report_lines.append("")
    report_lines.append("| Feature | PhiUSIIL Legit (0) | PhiUSIIL Phishing (1) | Wangchuk Legit (0) | Wangchuk Phishing (1) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for feature in FEATURE_ORDER:
        is_binary = feature in ["is_https", "has_ip"]
        phi_legit = format_compact_cell(phi_stats[feature].get(0, {}), is_binary)
        phi_phish = format_compact_cell(phi_stats[feature].get(1, {}), is_binary)
        wang_legit = format_compact_cell(wang_stats[feature].get(0, {}), is_binary)
        wang_phish = format_compact_cell(wang_stats[feature].get(1, {}), is_binary)
        
        feature_display = feature.replace("_", " ").title()
        report_lines.append(f"| **{feature_display}** | {phi_legit} | {phi_phish} | {wang_legit} | {wang_phish} |")
        
    report_lines.append("\n## Table 2: Detailed Stats (Flat)")
    report_lines.append("")
    report_lines.append("| Feature | Metric | PhiUSIIL Legit (0) | PhiUSIIL Phishing (1) | Wangchuk Legit (0) | Wangchuk Phishing (1) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    metrics = ["mean", "std", "min", "p25", "median", "p75", "p90", "p95", "max"]
    for feature in FEATURE_ORDER:
        is_binary = feature in ["is_https", "has_ip"]
        feature_display = feature.replace("_", " ").title()
        if is_binary:
            phi_val_0 = f"{phi_stats[feature][0]['pct_1']:.4f}%"
            phi_val_1 = f"{phi_stats[feature][1]['pct_1']:.4f}%"
            wang_val_0 = f"{wang_stats[feature][0]['pct_1']:.4f}%"
            wang_val_1 = f"{wang_stats[feature][1]['pct_1']:.4f}%"
            report_lines.append(f"| {feature_display} | % = 1 | {phi_val_0} | {phi_val_1} | {wang_val_0} | {wang_val_1} |")
        else:
            for metric in metrics:
                val_phi_0 = f"{phi_stats[feature][0][metric]:.4f}"
                val_phi_1 = f"{phi_stats[feature][1][metric]:.4f}"
                val_wang_0 = f"{wang_stats[feature][0][metric]:.4f}"
                val_wang_1 = f"{wang_stats[feature][1][metric]:.4f}"
                report_lines.append(f"| {feature_display} | {metric} | {val_phi_0} | {val_phi_1} | {val_wang_0} | {val_wang_1} |")
                
    # Class composition analysis
    report_lines.append("\n## Table 3: Class Composition Comparison")
    report_lines.append("")
    report_lines.append("| Dataset | Legitimate (0) | Phishing (1) | Total | % Phishing |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    
    phi_0 = int((phi_df["label"] == 0).sum())
    phi_1 = int((phi_df["label"] == 1).sum())
    phi_tot = len(phi_df)
    phi_pct = (phi_1 / phi_tot) * 100
    
    wang_0 = int((wang_df["label"] == 0).sum())
    wang_1 = int((wang_df["label"] == 1).sum())
    wang_tot = len(wang_df)
    wang_pct = (wang_1 / wang_tot) * 100
    
    report_lines.append(f"| PhiUSIIL | {phi_0:,} | {phi_1:,} | {phi_tot:,} | {phi_pct:.2f}% |")
    report_lines.append(f"| Wangchuk | {wang_0:,} | {wang_1:,} | {wang_tot:,} | {wang_pct:.2f}% |")
    
    # Save file
    report_content = "\n".join(report_lines)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Report successfully saved to {report_path}")

if __name__ == "__main__":
    main()
