"""Analyze probability calibration on internal and external datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import FEATURE_NAMES, load_dataset, score_model


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Make the last bin inclusive of 1.0
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model probability calibration.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=backend_dir / "models" / "xgboost_phish.pkl",
        help="Path to trained XGBoost model pkl."
    )
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

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}. Run experiments first to save the champion.")
        sys.exit(1)

    print(f"Loading champion model from {args.model_path}...")
    model = joblib.load(args.model_path)

    print(f"Loading sample of {args.sample_size:,} rows from PhiUSIIL...")
    phi_df = load_dataset(args.dataset, args.sample_size, args.seed)
    phi_X = phi_df[FEATURE_NAMES]
    phi_y = phi_df["label"].to_numpy()

    print(f"Loading external dataset from {args.external_dataset}...")
    # Evaluate calibration on the full external dataset
    wang_df = load_dataset(args.external_dataset, None, args.seed)
    wang_X = wang_df[FEATURE_NAMES]
    wang_y = wang_df["label"].to_numpy()

    print("\nRunning calibration checks...")
    
    # Internal predictions
    phi_probs = score_model(model, phi_X)
    phi_brier = brier_score_loss(phi_y, phi_probs)
    phi_ece = compute_ece(phi_y, phi_probs)
    
    # External predictions
    wang_probs = score_model(model, wang_X)
    wang_brier = brier_score_loss(wang_y, wang_probs)
    wang_ece = compute_ece(wang_y, wang_probs)

    print("PhiUSIIL (Internal) calibration:")
    print(f"  Brier Score: {phi_brier:.5f}")
    print(f"  ECE (10 Bins): {phi_ece:.5%}")
    
    print("Wangchuk (External) calibration:")
    print(f"  Brier Score: {wang_brier:.5f}")
    print(f"  ECE (10 Bins): {wang_ece:.5%}")

    results = {
        "phiusiil_internal": {
            "brier_score": float(phi_brier),
            "ece_10_bins": float(phi_ece)
        },
        "wangchuk_external": {
            "brier_score": float(wang_brier),
            "ece_10_bins": float(wang_ece)
        }
    }

    output_path = args.output_dir / "calibration_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nCalibration metrics saved to {output_path}")

    # Generate markdown table
    md_lines = [
        "| Evaluation Cohort | Brier Score (Lower is Better) | Expected Calibration Error (ECE) |",
        "| :--- | :---: | :---: |",
        f"| PhiUSIIL (Internal Test) | {phi_brier:.5f} | {phi_ece*100:.2f}% |",
        f"| Wangchuk (External Dataset) | {wang_brier:.5f} | {wang_ece*100:.2f}% |"
    ]
    
    md_path = args.output_dir / "calibration_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
