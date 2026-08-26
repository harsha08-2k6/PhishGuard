"""Evaluate a rule-based baseline and profile domain-disjoint split counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from run_experiments import FEATURE_NAMES, load_dataset


def evaluate_rule_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Classify as phishing if: has_ip == 1 or num_subdomains >= 2 or suspicious_keywords > 0 or url_entropy > 4.0."""
    target = df["label"].to_numpy()
    
    # Heuristic rules matching the FastAPI endpoint
    cond_ip = df["has_ip"] == 1
    cond_sub = df["num_subdomains"] >= 2
    cond_key = df["suspicious_keywords"] > 0
    cond_ent = df["url_entropy"] > 4.0
    
    preds = (cond_ip | cond_sub | cond_key | cond_ent).astype(int)
    
    return {
        "precision": float(precision_score(target, preds, zero_division=0)),
        "recall": float(recall_score(target, preds, zero_division=0)),
        "f1": float(f1_score(target, preds, zero_division=0))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule baseline and domain disjoint counts.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
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

    print("Loading full datasets...")
    phi_df = load_dataset(args.dataset, None, args.seed)
    wang_df = load_dataset(args.external_dataset, None, args.seed)

    print("\nEvaluating heuristic rule-based baseline...")
    phi_metrics = evaluate_rule_baseline(phi_df)
    wang_metrics = evaluate_rule_baseline(wang_df)

    print("PhiUSIIL (Internal) Rule Baseline:")
    print(f"  Precision: {phi_metrics['precision']:.4%}")
    print(f"  Recall:    {phi_metrics['recall']:.4%}")
    print(f"  F1-Score:  {phi_metrics['f1']:.4%}")
    
    print("Wangchuk (External) Rule Baseline:")
    print(f"  Precision: {wang_metrics['precision']:.4%}")
    print(f"  Recall:    {wang_metrics['recall']:.4%}")
    print(f"  F1-Score:  {wang_metrics['f1']:.4%}")

    print("\nProfiling domain-disjoint evaluation split...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(
        splitter.split(phi_df[FEATURE_NAMES], phi_df["label"], groups=phi_df["Domain"])
    )

    phi_train = phi_df.iloc[train_idx]
    phi_test = phi_df.iloc[test_idx]

    train_urls = len(phi_train)
    train_domains = int(phi_train["Domain"].nunique())
    train_phish = int((phi_train["label"] == 1).sum())
    train_legit = int((phi_train["label"] == 0).sum())

    test_urls = len(phi_test)
    test_domains = int(phi_test["Domain"].nunique())
    test_phish = int((phi_test["label"] == 1).sum())
    test_legit = int((phi_test["label"] == 0).sum())

    print("Domain-Disjoint Splits Summary:")
    print(f"  Training Split: {train_urls:,} URLs | {train_domains:,} unique domains")
    print(f"    (Legit: {train_legit:,}, Phish: {train_phish:,})")
    print(f"  Testing Split:  {test_urls:,} URLs | {test_domains:,} unique domains")
    print(f"    (Legit: {test_legit:,}, Phish: {test_phish:,})")

    # Verify that domains are strictly disjoint (intersection is empty)
    intersection = set(phi_train["Domain"]).intersection(set(phi_test["Domain"]))
    print(f"  Overlapping domains count: {len(intersection)}")

    results = {
        "rule_baseline": {
            "phiusiil_internal": phi_metrics,
            "wangchuk_external": wang_metrics
        },
        "domain_disjoint_split_profile": {
            "training": {
                "urls": train_urls,
                "unique_domains": train_domains,
                "legitimate": train_legit,
                "phishing": train_phish
            },
            "testing": {
                "urls": test_urls,
                "unique_domains": test_domains,
                "legitimate": test_legit,
                "phishing": test_phish
            },
            "domain_overlap": len(intersection)
        }
    }

    output_path = args.output_dir / "baseline_and_domain_stats.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults successfully saved to {output_path}")

    # Generate markdown table for domain splits
    md_lines = [
        "| Split Cohort | Total URLs | Unique Domains | Legitimate (0) | Phishing (1) | Phishing Share |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| Training Set (80%) | {train_urls:,} | {train_domains:,} | {train_legit:,} | {train_phish:,} | {(train_phish/train_urls)*100:.2f}% |",
        f"| Testing Set (20%) | {test_urls:,} | {test_domains:,} | {test_legit:,} | {test_phish:,} | {(test_phish/test_urls)*100:.2f}% |"
    ]
    md_path = args.output_dir / "domain_split_profile_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
