"""Formal Feature-Selection Audit using Mutual Information and Correlation Analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Set up path to import run_experiments
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import load_dataset, FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mutual Information and Correlation audit.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample of {args.sample_size:,} rows from dataset...")
    df = load_dataset(args.dataset, args.sample_size, args.seed)
    
    X = df[FEATURE_NAMES]
    y = df["label"].to_numpy()

    print("Calculating Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=args.seed)
    mi_dict = dict(zip(FEATURE_NAMES, map(float, mi_scores)))
    
    # Sort features by MI score descending
    sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    print("\nMutual Information Scores:")
    for feat, score in sorted_mi:
        print(f"  {feat:<25}: {score:.5f}")

    print("\nCalculating Correlation Matrix...")
    corr_matrix = X.corr(method="pearson").abs()
    
    # Extract top redundant pairs
    redundant_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            f1, f2 = FEATURE_NAMES[i], FEATURE_NAMES[j]
            r = corr_matrix.loc[f1, f2]
            redundant_pairs.append((f1, f2, float(r)))
            
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop Feature Correlations:")
    for f1, f2, r in redundant_pairs[:10]:
        print(f"  {f1} <-> {f2}: r = {r:.5f}")

    results = {
        "mutual_information": mi_dict,
        "sorted_mutual_information": sorted_mi,
        "correlations": redundant_pairs
    }

    output_path = args.output_dir / "feature_selection_audit.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAudit results successfully saved to {output_path}")


if __name__ == "__main__":
    main()
