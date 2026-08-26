"""Analyze dataset-artifact sensitivity and feature dimensionality tradeoffs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import FEATURE_NAMES, load_dataset, score_model


def get_xgb_classifier(seed: int):
    xgboost = importlib.import_module("xgboost")
    return xgboost.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


def evaluate_config(
    features_list: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    cv_folds: int
) -> tuple[float, float, float]:
    """Run CV on train_df and test on test_df, return CV F1, Ext F1, and Inf Latency (ms/URL)."""
    # 1. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_f1s = []
    
    X_train_full = train_df[features_list]
    y_train_full = train_df["label"].to_numpy()
    
    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        clf = get_xgb_classifier(seed).fit(X_tr, y_tr)
        probs = score_model(clf, X_val)
        preds = (probs >= 0.5).astype(int)
        cv_f1s.append(f1_score(y_val, preds, zero_division=0))
        
    mean_cv_f1 = float(np.mean(cv_f1s))

    # 2. Train on full train_df, test on test_df
    clf_full = get_xgb_classifier(seed).fit(X_train_full, y_train_full)
    X_test_full = test_df[features_list]
    y_test_full = test_df["label"].to_numpy()
    
    # Measure inference latency (run 3 times to stabilize)
    t_start = time.perf_counter()
    for _ in range(3):
        ext_probs = score_model(clf_full, X_test_full)
    t_end = time.perf_counter()
    inf_latency_ms = ((t_end - t_start) / (len(test_df) * 3)) * 1000.0
    
    ext_preds = (ext_probs >= 0.5).astype(int)
    ext_f1 = float(f1_score(y_test_full, ext_preds, zero_division=0))

    return mean_cv_f1, ext_f1, inf_latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity and ablation tests.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument(
        "--external-dataset",
        type=Path,
        default=experiments_dir / "external_wangchuk.csv",
        help="Path to external Wangchuk dataset CSV."
    )
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample of {args.sample_size:,} rows from PhiUSIIL...")
    train_df = load_dataset(args.dataset, args.sample_size, args.seed)
    
    print("Loading full external Wangchuk dataset...")
    test_df = load_dataset(args.external_dataset, None, args.seed)

    # Deployed 12
    configs = {
        "Full Deployed (12 features)": FEATURE_NAMES,
        
        # Feature subsets
        "Top-7 Features (MI-based)": [
            "num_slashes", "is_https", "num_digits", "url_length", "url_entropy", "special_chars", "domain_length"
        ],
        "Top-5 Features (MI-based)": [
            "num_slashes", "is_https", "num_digits", "url_length", "url_entropy"
        ],
        
        # Shortcut ablations
        "Remove HTTPS Shortcut (11 features)": [f for f in FEATURE_NAMES if f != "is_https"],
        "Remove IP Address Indicator (11 features)": [f for f in FEATURE_NAMES if f != "has_ip"],
    }

    results = []

    print("\nEvaluating configurations...")
    for config_name, features_list in configs.items():
        print(f"Running: {config_name} ({len(features_list)} features)...")
        cv_f1, ext_f1, inf_latency = evaluate_config(
            features_list, train_df, test_df, args.seed, args.cv_folds
        )
        results.append({
            "configuration": config_name,
            "feature_count": len(features_list),
            "features": features_list,
            "cv_f1": cv_f1,
            "external_f1": ext_f1,
            "inference_latency_ms": inf_latency
        })
        print(f"  CV F1: {cv_f1:.4%} | External F1: {ext_f1:.4%} | Latency: {inf_latency:.4f} ms/URL")

    output_path = args.output_dir / "sensitivity_ablation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAblation metrics saved to {output_path}")

    # Generate markdown table
    md_lines = [
        "| Feature Configuration | Count | In-Dataset F1 (PhiUSIIL CV) | External F1 (Wangchuk) | Inference Latency (ms/URL) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for res in results:
        md_lines.append(
            f"| {res['configuration']} | {res['feature_count']} | {res['cv_f1']*100:.2f}% | {res['external_f1']*100:.2f}% | {res['inference_latency_ms']:.5f} ms |"
        )

    md_path = args.output_dir / "sensitivity_ablation_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
