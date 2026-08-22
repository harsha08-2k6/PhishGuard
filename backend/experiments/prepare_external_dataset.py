"""Prepare the Tandin Wangchuk (Mendeley 2026) phishing URL dataset.

The Mendeley archive (doi: 10.17632/3jddhy2f6s/1) contains two files:
    phishing_urls.csv   - one URL per row, no header  OR  header 'url'
    legitimate_urls.csv - one URL per row, no header  OR  header 'url'

This script merges them into a single URL,label CSV compatible with the
PhishGuard experiment pipeline (0 = legitimate, 1 = phishing).

Usage
-----
    python experiments/prepare_external_dataset.py \
        --phishing    path/to/phishing_urls.csv \
        --legitimate  path/to/legitimate_urls.csv \
        --output      experiments/external_wangchuk.csv

Then run the cross-dataset experiment:
    python experiments/run_experiments.py \
        --dataset     path/to/PhiUSIIL_Phishing_URL_Dataset.csv \
        --output-dir  experiments/results-external \
        --external-dataset experiments/external_wangchuk.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_url_file(path: Path) -> pd.Series:
    """Load a single-column URL file with or without a header row."""
    raw = pd.read_csv(path, header=None, names=["url"], dtype=str)
    # If the first row looks like a header (non-URL text), drop it.
    first = str(raw.iloc[0, 0]).strip().lower()
    if first in {"url", "urls", "link", "links"} or not (
        first.startswith("http") or "." in first
    ):
        raw = raw.iloc[1:].reset_index(drop=True)
    return raw["url"].str.strip().dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phishing", type=Path, required=True,
                        help="Path to phishing_urls.csv from the Wangchuk dataset.")
    parser.add_argument("--legitimate", type=Path, required=True,
                        help="Path to legitimate_urls.csv from the Wangchuk dataset.")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/external_wangchuk.csv"))
    args = parser.parse_args()

    phishing_urls = _load_url_file(args.phishing)
    legitimate_urls = _load_url_file(args.legitimate)

    combined = pd.concat([
        pd.DataFrame({"URL": phishing_urls, "label": 1}),
        pd.DataFrame({"URL": legitimate_urls, "label": 0}),
    ], ignore_index=True)

    # Drop blank or duplicate URLs.
    before = len(combined)
    combined = combined[combined["URL"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    removed = before - len(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    phishing_count = int((combined["label"] == 1).sum())
    legitimate_count = int((combined["label"] == 0).sum())
    print(f"Wangchuk dataset prepared:")
    print(f"  Phishing  : {phishing_count:>7,}")
    print(f"  Legitimate: {legitimate_count:>7,}")
    print(f"  Total     : {len(combined):>7,}  ({removed} duplicates removed)")
    print(f"  Saved to  : {args.output}")


if __name__ == "__main__":
    main()
