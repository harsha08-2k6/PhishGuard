"""Audit URL-only PhiUSIIL features for leakage and split risks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_features import FEATURE_ORDER, extract_features  # noqa: E402


def label_purity(frame: pd.DataFrame, group_column: str) -> dict[str, float | int]:
    grouped = frame.groupby(group_column)["label"]
    counts = grouped.nunique()
    mixed_groups = int((counts > 1).sum())
    total_groups = int(counts.size)
    return {
        "groups": total_groups,
        "mixed_label_groups": mixed_groups,
        "mixed_group_rate": mixed_groups / total_groups if total_groups else 0.0,
    }


def conditional_purity(frame: pd.DataFrame, feature: str) -> dict[str, float | int]:
    grouped = frame.groupby(feature)["label"]
    sizes = grouped.size()
    majority = grouped.value_counts().groupby(level=0).max()
    weighted_purity = float(majority.sum() / sizes.sum())
    return {
        "unique_values": int(sizes.size),
        "weighted_label_purity": weighted_purity,
        "max_value_group_size": int(sizes.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/leakage_audit.json"))
    args = parser.parse_args()

    source_columns = pd.read_csv(args.dataset, nrows=0).columns.tolist()
    frame = pd.read_csv(args.dataset, usecols=["URL", "Domain", "label"], low_memory=False)
    frame["URL"] = frame["URL"].astype(str)
    frame["Domain"] = frame["Domain"].astype(str)
    frame["normalized_url"] = frame["URL"].map(
        lambda value: value.strip().lower()
    )
    frame["derived_domain"] = frame["URL"].map(
        lambda value: urlparse(value if "://" in value else f"http://{value}").netloc.lower()
    )
    features = pd.DataFrame([extract_features(value) for value in frame["URL"]])
    audit_frame = pd.concat([frame.reset_index(drop=True), features], axis=1)

    numeric_correlations = (
        audit_frame[FEATURE_ORDER + ["label"]]
        .corr(numeric_only=True)["label"]
        .drop("label")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    url_label_counts = audit_frame.groupby("normalized_url")["label"].nunique()
    domain_label_counts = audit_frame.groupby("Domain")["label"].nunique()
    url_group_sizes = audit_frame.groupby("normalized_url").size()
    domain_group_sizes = audit_frame.groupby("Domain").size()
    feature_purity = {
        feature: conditional_purity(audit_frame, feature) for feature in FEATURE_ORDER
    }

    result = {
        "dataset": str(args.dataset),
        "rows": int(len(audit_frame)),
        "source_columns": source_columns,
        "model_features": FEATURE_ORDER,
        "excluded_identifier_columns": ["FILENAME"],
        "excluded_page_content_columns": [
            column for column in source_columns
            if column in {"LineOfCode", "LargestLineLength", "Title", "HasTitle", "HasPasswordField", "NoOfExternalRef"}
        ],
        "label_distribution": {
            str(key): int(value) for key, value in audit_frame["label"].value_counts().sort_index().items()
        },
        "row_quality": {
            "missing_cells_in_used_columns": int(audit_frame[["URL", "Domain", "label"]].isna().sum().sum()),
            "exact_duplicate_rows_in_used_columns": int(audit_frame[["URL", "Domain", "label"]].duplicated().sum()),
            "blank_urls": int((audit_frame["normalized_url"] == "").sum()),
        },
        "group_overlap": {
            "normalized_url": label_purity(audit_frame, "normalized_url"),
            "dataset_domain": label_purity(audit_frame, "Domain"),
            "derived_domain": label_purity(audit_frame, "derived_domain"),
            "repeated_url_groups": int((url_group_sizes > 1).sum()),
            "mixed_label_url_groups": int((url_label_counts > 1).sum()),
            "repeated_domain_groups": int((domain_group_sizes > 1).sum()),
        },
        "feature_label_correlations": {
            key: float(value) for key, value in numeric_correlations.items()
        },
        "feature_value_purity": feature_purity,
        "special_checks": {
            "https_label_table": {
                str(key): {str(label): int(count) for label, count in values.items()}
                for key, values in audit_frame.groupby("is_https")["label"].value_counts().groupby(level=0)
            },
            "ip_label_table": {
                str(key): {str(label): int(count) for label, count in values.items()}
                for key, values in audit_frame.groupby("has_ip")["label"].value_counts().groupby(level=0)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": result["rows"],
        "mixed_url_groups": result["group_overlap"]["normalized_url"]["mixed_label_groups"],
        "mixed_domain_groups": result["group_overlap"]["dataset_domain"]["mixed_label_groups"],
        "top_absolute_correlations": sorted(
            result["feature_label_correlations"].items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
