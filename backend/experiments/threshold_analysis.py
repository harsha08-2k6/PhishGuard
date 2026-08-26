"""Evaluate classification performance at different decision thresholds on the external dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import load_dataset, FEATURE_NAMES, score_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate threshold trade-offs.")
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
    external_df = load_dataset(args.external_dataset, None, args.seed)
    
    print(f"Loading champion model from {args.model_path}...")
    model = joblib.load(args.model_path)

    print("Computing prediction probabilities...")
    probabilities = score_model(model, external_df[FEATURE_NAMES])
    target = external_df["label"].to_numpy()

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = []

    print("\nThreshold Evaluation Metrics:")
    print(f"  {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        precision = precision_score(target, preds, zero_division=0)
        recall = recall_score(target, preds, zero_division=0)
        f1 = f1_score(target, preds, zero_division=0)
        
        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        })
        print(f"  {t:<10.2f} | {precision:<10.4%} | {recall:<10.4%} | {f1:<10.4%}")

    df_results = pd.DataFrame(results)
    output_path = args.output_dir / "threshold_analysis.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nThreshold analysis results saved to {output_path}")

    # Generate markdown table for direct copying
    md_lines = [
        "| Decision Threshold | Precision | Recall | F1-score |",
        "| :--- | :---: | :---: | :---: |"
    ]
    for res in results:
        md_lines.append(
            f"| {res['threshold']:.2f} | {res['precision']:.2f}% | {res['recall']:.2f}% | {res['f1']:.2f}% |"
        )
    
    # Scale percentages by 100 for display
    md_lines_pct = [
        "| Decision Threshold | Precision | Recall | F1-score |",
        "| :--- | :---: | :---: | :---: |"
    ]
    for res in results:
        md_lines_pct.append(
            f"| {res['threshold']:.2f} | {res['precision']*100:.2f}% | {res['recall']*100:.2f}% | {res['f1']*100:.2f}% |"
        )

    md_path = args.output_dir / "threshold_analysis_table.md"
    md_path.write_text("\n".join(md_lines_pct), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
