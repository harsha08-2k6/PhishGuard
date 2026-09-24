"""Measure model training and prediction efficiency metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone

# Set up paths
experiments_dir = Path(__file__).resolve().parent
backend_dir = experiments_dir.parent
sys.path.insert(0, str(experiments_dir))
sys.path.insert(0, str(backend_dir))

from extract_features import extract_features
from run_experiments import load_dataset, FEATURE_NAMES, make_models, score_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure computational efficiency.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to PhiUSIIL dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=experiments_dir / "results-external")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading subset of URLs to measure extraction latency...")
    # Load raw URLs first to measure feature extraction time
    raw_df = pd.read_csv(args.dataset, usecols=["URL"], nrows=args.sample_size)
    urls = raw_df["URL"].astype(str).tolist()

    print("Measuring feature extraction latency...")
    t_start = time.perf_counter()
    for url in urls:
        _ = extract_features(url)
    t_end = time.perf_counter()
    extract_latency_ms = ((t_end - t_start) / len(urls)) * 1000.0
    print(f"  Feature extraction: {extract_latency_ms:.4f} ms per URL")

    # Load dataset with extracted features for training/inference benchmark
    print("\nLoading dataset with extracted features...")
    df = load_dataset(args.dataset, args.sample_size, args.seed)
    X = df[FEATURE_NAMES]
    y = df["label"].to_numpy()

    # Recreate the 5 models
    models = make_models(args.seed, skip_xgboost=False)
    
    # We will measure relative training time on this sample size (e.g. 10k rows)
    train_times = {}
    inference_times_ms = {}
    model_sizes_kb = {}

    print("\nBenchmarking models...")
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Benchmark training time
        t_start = time.perf_counter()
        fitted = clone(model).fit(X, y)
        t_end = time.perf_counter()
        train_time = t_end - t_start
        train_times[model_name] = train_time
        print(f"  Training time (on {args.sample_size} rows): {train_time:.4f} seconds")

        # Benchmark inference time (run prediction multiple times on the set to get stable reading)
        t_start = time.perf_counter()
        for _ in range(5):
            _ = score_model(fitted, X)
        t_end = time.perf_counter()
        inference_latency_ms = ((t_end - t_start) / (len(df) * 5)) * 1000.0
        inference_times_ms[model_name] = inference_latency_ms
        print(f"  Inference latency: {inference_latency_ms:.4f} ms per URL")

        # Save a temporary model to check serialized file size
        temp_path = args.output_dir / "temp_model.pkl"
        joblib.dump(fitted, temp_path)
        size_kb = os.path.getsize(temp_path) / 1024.0
        model_sizes_kb[model_name] = size_kb
        print(f"  Serialized size: {size_kb:.2f} KB")
        if temp_path.exists():
            temp_path.unlink()

    results = {
        "feature_extraction_latency_ms": extract_latency_ms,
        "sample_size": args.sample_size,
        "training_time_seconds": train_times,
        "inference_latency_ms": inference_times_ms,
        "model_size_kb": model_sizes_kb
    }

    output_path = args.output_dir / "efficiency_metrics.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEfficiency metrics saved to {output_path}")

    # Generate markdown table
    md_lines = [
        "| Model | Model Size (KB) | Training Time (10k rows) | Inference Latency (per URL) | Total Latency (Extraction + Model) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for model_name in models:
        size = model_sizes_kb[model_name]
        tr_time = train_times[model_name]
        inf_lat = inference_times_ms[model_name]
        tot_lat = extract_latency_ms + inf_lat
        md_lines.append(
            f"| {model_name} | {size:.2f} KB | {tr_time:.3f} s | {inf_lat:.4f} ms | {tot_lat:.4f} ms |"
        )

    md_path = args.output_dir / "efficiency_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
