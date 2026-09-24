"""Run repeated stratified 5-fold cross-validation with 95% Confidence Intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import FEATURE_NAMES, load_dataset, make_models, score_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated cross-validation with CIs.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sample of {args.sample_size:,} rows from dataset...")
    df = load_dataset(args.dataset, args.sample_size, args.seed)
    X = df[FEATURE_NAMES]
    y = df["label"].to_numpy()

    # Recreate the 5 models
    models = make_models(args.seed, skip_xgboost=False)

    rskf = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed
    )

    results = {}

    print(f"\nRunning {args.n_repeats} x {args.n_splits}-fold CV...")
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        f1_scores = []
        
        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fitted = clone(model).fit(X_train, y_train)
            probs = score_model(fitted, X_test)
            preds = (probs >= 0.5).astype(int)
            
            # Compute F1
            from sklearn.metrics import f1_score
            f1_scores.append(float(f1_score(y_test, preds, zero_division=0)))
            
        f1_scores = np.array(f1_scores)
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores, ddof=1))
        
        # Calculate 95% Confidence Interval using t-distribution
        dof = len(f1_scores) - 1
        t_crit = t.ppf(0.975, df=dof)
        sem = std_f1 / np.sqrt(len(f1_scores))
        ci_half = t_crit * sem
        ci_lower = mean_f1 - ci_half
        ci_upper = mean_f1 + ci_half
        
        results[model_name] = {
            "mean": mean_f1,
            "std": std_f1,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_scores": f1_scores.tolist()
        }
        print(f"  Mean F1: {mean_f1:.4%} | Std F1: {std_f1:.4%} | 95% CI: [{ci_lower:.4%}, {ci_upper:.4%}]")

    output_path = args.output_dir / "repeated_cv_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRepeated CV results saved to {output_path}")

    # Generate markdown table
    md_lines = [
        "| Model | F1 Mean | F1 Std | 95% Confidence Interval |",
        "| :--- | :---: | :---: | :---: |"
    ]
    for model_name in models:
        r = results[model_name]
        md_lines.append(
            f"| {model_name} | {r['mean']*100:.2f}% | ±{r['std']*100:.2f}% | [{r['ci_lower']*100:.2f}%, {r['ci_upper']*100:.2f}%] |"
        )
    
    md_path = args.output_dir / "repeated_cv_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
