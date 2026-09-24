"""Run detailed error analysis of predictions on the external dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import load_dataset, FEATURE_NAMES, score_model


def obfuscate_url(url: str, is_phishing: bool, index: int) -> str:
    """Make the URL safe for academic publication by obfuscating the domain."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        scheme = parsed.scheme if parsed.scheme else "http"
        domain_label = f"phish-domain-{index}.com" if is_phishing else f"legit-domain-{index}.org"
        path = parsed.path
        if parsed.query:
            path += f"?{parsed.query[:15]}..."
        return f"{scheme}://{domain_label}{path[:30]}"
    except Exception:
        return f"http://{'phish' if is_phishing else 'legit'}-obfuscated-{index}.com/path"


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform error analysis on the external dataset.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=backend_dir / "models" / "xgboost_phish.pkl",
        help="Path to trained model pkl."
    )
    parser.add_argument(
        "--external-dataset",
        type=Path,
        default=experiments_dir / "external_wangchuk.csv",
        help="Path to external Wangchuk dataset CSV."
    )
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}. Run experiments first to save the champion.")
        sys.exit(1)

    print(f"Loading external dataset from {args.external_dataset}...")
    df = load_dataset(args.external_dataset, None, args.seed)
    
    print(f"Loading champion model from {args.model_path}...")
    model = joblib.load(args.model_path)

    print("Running predictions...")
    probabilities = score_model(model, df[FEATURE_NAMES])
    predictions = (probabilities >= 0.5).astype(int)
    labels = df["label"].to_numpy()

    df["prob"] = probabilities
    df["pred"] = predictions

    # Identify FP, FN, TP, TN
    fp_mask = (labels == 0) & (predictions == 1)
    fn_mask = (labels == 1) & (predictions == 0)
    tp_mask = (labels == 1) & (predictions == 1)
    tn_mask = (labels == 0) & (predictions == 0)

    fps = df[fp_mask].reset_index(drop=True)
    fns = df[fn_mask].reset_index(drop=True)
    tps = df[tp_mask].reset_index(drop=True)
    tns = df[tn_mask].reset_index(drop=True)

    print(f"\nPrediction counts:")
    print(f"  True Negatives (TN):  {len(tns):,}")
    print(f"  False Positives (FP): {len(fps):,}")
    print(f"  False Negatives (FN): {len(fns):,}")
    print(f"  True Positives (TP):  {len(tps):,}")

    # Compute feature averages
    averages = {}
    for feature in FEATURE_NAMES:
        averages[feature] = {
            "TN": float(tns[feature].mean()) if len(tns) > 0 else 0.0,
            "FP": float(fps[feature].mean()) if len(fps) > 0 else 0.0,
            "FN": float(fns[feature].mean()) if len(fns) > 0 else 0.0,
            "TP": float(tps[feature].mean()) if len(tps) > 0 else 0.0,
        }

    print("\nFeature Averages by Confusion Matrix Category:")
    print(f"  {'Feature':<20} | {'TN':<10} | {'FP':<10} | {'FN':<10} | {'TP':<10}")
    print("-" * 70)
    for feat in FEATURE_NAMES:
        v = averages[feat]
        print(f"  {feat:<20} | {v['TN']:<10.4f} | {v['FP']:<10.4f} | {v['FN']:<10.4f} | {v['TP']:<10.4f}")

    # Extract 10 representative False Negatives
    representative_fns = []
    sample_fns = fns.sample(n=min(10, len(fns)), random_state=args.seed).reset_index(drop=True)
    for idx, row in sample_fns.iterrows():
        obf = obfuscate_url(row["URL"], is_phishing=True, index=idx+1)
        representative_fns.append({
            "obfuscated_url": obf,
            "probability": float(row["prob"]),
            **{feat: float(row[feat]) for feat in FEATURE_NAMES}
        })

    # Extract 10 representative False Positives
    representative_fps = []
    sample_fps = fps.sample(n=min(10, len(fps)), random_state=args.seed).reset_index(drop=True)
    for idx, row in sample_fps.iterrows():
        obf = obfuscate_url(row["URL"], is_phishing=False, index=idx+1)
        representative_fps.append({
            "obfuscated_url": obf,
            "probability": float(row["prob"]),
            **{feat: float(row[feat]) for feat in FEATURE_NAMES}
        })

    results = {
        "metrics_summary": {
            "tn": len(tns),
            "fp": len(fps),
            "fn": len(fns),
            "tp": len(tps)
        },
        "feature_averages": averages,
        "representative_false_negatives": representative_fns,
        "representative_false_positives": representative_fps
    }

    output_path = args.output_dir / "error_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nError analysis results saved to {output_path}")

    # Write out a markdown summary report
    md_lines = []
    md_lines.append("# Error Analysis Report: External Dataset Failures")
    md_lines.append("\n## Table 1: Feature Averages Across Confusion Matrix Quadrants")
    md_lines.append("")
    md_lines.append("| Feature | True Legit (TN) | False Phish (FP) | False Legit (FN) | True Phish (TP) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for feat in FEATURE_NAMES:
        v = averages[feat]
        md_lines.append(f"| {feat.replace('_', ' ').title()} | {v['TN']:.4f} | {v['FP']:.4f} | {v['FN']:.4f} | {v['TP']:.4f} |")

    md_lines.append("\n## Table 2: Representative False Negatives (Phishing Predicted as Legit)")
    md_lines.append("\nURLs are obfuscated for security.")
    md_lines.append("")
    md_lines.append("| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for fn in representative_fns:
        md_lines.append(
            f"| `{fn['obfuscated_url']}` | {fn['probability']*100:.1f}% | {int(fn['is_https'])} | {int(fn['url_length'])} | {int(fn['num_dots'])} | {int(fn['num_digits'])} | {int(fn['special_chars'])} | {int(fn['suspicious_keywords'])} | {fn['url_entropy']:.3f} |"
        )

    md_lines.append("\n## Table 3: Representative False Positives (Legit Predicted as Phishing)")
    md_lines.append("\nURLs are obfuscated for security.")
    md_lines.append("")
    md_lines.append("| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for fp in representative_fps:
        md_lines.append(
            f"| `{fp['obfuscated_url']}` | {fp['probability']*100:.1f}% | {int(fp['is_https'])} | {int(fp['url_length'])} | {int(fp['num_dots'])} | {int(fp['num_digits'])} | {int(fp['special_chars'])} | {int(fp['suspicious_keywords'])} | {fp['url_entropy']:.3f} |"
        )

    md_report_path = args.output_dir / "error_analysis_report.md"
    md_report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Error analysis markdown report saved to {md_report_path}")


if __name__ == "__main__":
    main()
