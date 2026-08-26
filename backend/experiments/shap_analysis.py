"""Run SHAP explainability analysis on the trained XGBoost model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import load_dataset, FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP explainability analysis.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=backend_dir / "models" / "xgboost_phish.pkl",
        help="Path to trained XGBoost model pkl."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}. Run experiments first to save the champion.")
        sys.exit(1)

    try:
        import shap
    except ImportError:
        print("Error: shap library is not installed. Install it with `pip install shap`.")
        sys.exit(1)

    print(f"Loading sample of {args.sample_size:,} rows from PhiUSIIL...")
    df = load_dataset(args.dataset, args.sample_size, args.seed)
    X = df[FEATURE_NAMES]

    print(f"Loading XGBoost model from {args.model_path}...")
    model = joblib.load(args.model_path)

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # SHAP values can sometimes return a list of arrays for multi-class, or a single array for binary.
    # For binary classification with XGBoost, it usually returns a single 2D array of shape (samples, features).
    shap_output = explainer.shap_values(X)
    
    if isinstance(shap_output, list):
        # If list, take the positive class contributions (usually index 1)
        shap_values = shap_output[1] if len(shap_output) > 1 else shap_output[0]
    else:
        shap_values = shap_output

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = dict(zip(FEATURE_NAMES, map(float, mean_abs_shap)))
    
    # Sort features by SHAP importance descending
    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nSHAP Feature Importance (Mean Absolute SHAP Value):")
    for feat, val in sorted_shap:
        print(f"  {feat:<25}: {val:.5f}")

    results = {
        "mean_absolute_shap": shap_importance,
        "sorted_mean_absolute_shap": sorted_shap
    }

    output_path = args.output_dir / "shap_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSHAP analysis results saved to {output_path}")

    # Generate markdown table for direct copying
    md_lines = [
        "| Feature | Mean Absolute SHAP Value (Log-Odds Impact) | Ranking |",
        "| :--- | :---: | :---: |"
    ]
    for idx, (feat, val) in enumerate(sorted_shap, 1):
        md_lines.append(
            f"| {feat.replace('_', ' ').title()} | {val:.5f} | #{idx} |"
        )
    
    md_path = args.output_dir / "shap_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")

    # Try saving a summary plot if matplotlib is installed
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plot_path = args.output_dir / "xgboost_shap_summary.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"SHAP summary plot successfully saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save SHAP summary plot due to: {e}")


if __name__ == "__main__":
    main()
