"""Reproducible URL-only PhiUSIIL experiments.

The default run evaluates five classifiers on the same 12 features produced by
backend.extract_features. It reports stratified cross-validation, a domain-
disjoint holdout, and a champion ablation study.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_features import extract_features  # noqa: E402


FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_subdomains",
    "num_digits",
    "special_chars",
    "num_hyphens",
    "num_slashes",
    "has_ip",
    "is_https",
    "suspicious_keywords",
    "url_entropy",
]

ABLATION_GROUPS = {
    "all_12": FEATURE_NAMES,
    "structure": [
        "url_length",
        "domain_length",
        "num_dots",
        "num_subdomains",
        "num_hyphens",
        "num_slashes",
    ],
    "security": ["has_ip", "is_https", "suspicious_keywords"],
    "randomness": ["num_digits", "special_chars", "url_entropy"],
}

CUMULATIVE_ABLATION_GROUPS = {
    "lexical": ["url_length", "num_digits", "special_chars"],
    "lexical_structural": [
        "url_length", "num_digits", "special_chars",
        "num_dots", "num_subdomains", "num_hyphens", "num_slashes",
    ],
    "plus_domain": [
        "url_length", "num_digits", "special_chars",
        "num_dots", "num_subdomains", "num_hyphens", "num_slashes",
        "domain_length",
    ],
    "plus_security": [
        "url_length", "num_digits", "special_chars",
        "num_dots", "num_subdomains", "num_hyphens", "num_slashes",
        "has_ip", "is_https", "suspicious_keywords",
    ],
    "plus_entropy": [
        "url_length", "num_digits", "special_chars",
        "num_dots", "num_subdomains", "num_hyphens", "num_slashes",
        "has_ip", "is_https", "suspicious_keywords", "url_entropy",
    ],
}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def make_models(seed: int, skip_xgboost: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, class_weight="balanced", random_state=seed
        ),
        "SVM (RBF approximation)": Pipeline([
            ("scale", StandardScaler()),
            ("rbf_map", Nystroem(kernel="rbf", n_components=50, gamma=1.0, random_state=seed)),
            ("model", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
        ]),
    }

    if not skip_xgboost:
        try:
            xgboost = importlib.import_module("xgboost")
            XGBClassifier = xgboost.XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is required for the complete five-model experiment. "
                "Install it with `pip install xgboost`, or use --skip-xgboost for a smoke run."
            ) from error
        models["XGBoost"] = XGBClassifier(
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
    return models


def score_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    decision_values = model.decision_function(features)
    return 1.0 / (1.0 + np.exp(-np.clip(decision_values, -40, 40)))


def load_dataset(path: Path, sample_size: int | None, seed: int) -> pd.DataFrame:
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    required_columns = ["URL", "label"]
    if not set(required_columns).issubset(available_columns):
        raise ValueError("Dataset must contain URL and label columns for evaluation.")
    columns = required_columns + (["Domain"] if "Domain" in available_columns else [])
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["URL"] = frame["URL"].astype(str)
    if "Domain" not in frame:
        frame["Domain"] = frame["URL"].map(
            lambda url: urlparse(url if "://" in url else f"http://{url}").netloc
        )
    frame["Domain"] = frame["Domain"].astype(str)
    frame = frame.dropna(subset=["URL", "Domain", "label"])
    if sample_size and sample_size < len(frame):
        frame = frame.groupby("label", group_keys=False).sample(
            n=max(1, sample_size // 2), random_state=seed
        )
    feature_frame = pd.DataFrame(
        [extract_features(url) for url in frame["URL"]],
        index=frame.index,
    )
    return pd.concat([frame, feature_frame], axis=1).reset_index(drop=True)


def evaluate_cv(
    model_name: str,
    model: Any,
    data: pd.DataFrame,
    folds: int,
    seed: int,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics = []
    features = data[FEATURE_NAMES]
    target = data["label"].to_numpy()
    for train_index, test_index in splitter.split(features, target):
        fitted = clone(model).fit(features.iloc[train_index], target[train_index])
        probabilities = score_model(fitted, features.iloc[test_index])
        predictions = (probabilities >= 0.5).astype(int)
        fold_metrics.append(metric_row(target[test_index], predictions, probabilities))
    return {key: float(np.mean([row[key] for row in fold_metrics])) for key in fold_metrics[0]}


def evaluate_domain_holdout(
    model: Any,
    data: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_index, test_index = next(
        splitter.split(data[FEATURE_NAMES], data["label"], groups=data["Domain"])
    )
    target = data["label"].to_numpy()
    fitted = clone(model).fit(data.loc[train_index, FEATURE_NAMES], target[train_index])
    probabilities = score_model(fitted, data.loc[test_index, FEATURE_NAMES])
    predictions = (probabilities >= 0.5).astype(int)
    return metric_row(target[test_index], predictions, probabilities)


def run_ablation(
    model: Any,
    data: pd.DataFrame,
    folds: int,
    seed: int,
    groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    target = data["label"].to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for group_name, feature_names in groups.items():
        fold_metrics = []
        for train_index, test_index in splitter.split(data[feature_names], target):
            fitted = clone(model).fit(data.loc[train_index, feature_names], target[train_index])
            probabilities = score_model(fitted, data.loc[test_index, feature_names])
            predictions = (probabilities >= 0.5).astype(int)
            fold_metrics.append(metric_row(target[test_index], predictions, probabilities))
        rows.append({
            "feature_set": group_name,
            "features": feature_names,
            **{key: float(np.mean([row[key] for row in fold_metrics])) for key in fold_metrics[0]},
        })
    return rows


def evaluate_external(
    models: dict[str, Any],
    training_data: pd.DataFrame,
    external_path: Path,
    seed: int,
) -> list[dict[str, float | str]]:
    external = load_dataset(external_path, None, seed)
    target = external["label"].to_numpy()
    rows = []
    for model_name, model in models.items():
        fitted = clone(model).fit(training_data[FEATURE_NAMES], training_data["label"])
        probabilities = score_model(fitted, external[FEATURE_NAMES])
        predictions = (probabilities >= 0.5).astype(int)
        rows.append({"model": model_name, **metric_row(target, predictions, probabilities)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--external-dataset", type=Path)
    parser.add_argument("--save-all-models", action="store_true")
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.sample_size, args.seed)
    models = make_models(args.seed, args.skip_xgboost)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.artifacts_only:
        model_directory = Path(__file__).resolve().parents[1] / "models"
        model_directory.mkdir(parents=True, exist_ok=True)
        artifact_names = {
            "XGBoost": "xgboost_phish.pkl",
            "Random Forest": "random_forest_phish.pkl",
            "SVM (RBF approximation)": "svm_rbf_approximation.pkl",
            "Decision Tree": "decision_tree_phish.pkl",
            "Logistic Regression": "logistic_regression_phish.pkl",
        }
        for model_name, model in models.items():
            print(f"Training artifact {model_name}...")
            fitted_model = clone(model).fit(data[FEATURE_NAMES], data["label"])
            joblib.dump(fitted_model, model_directory / artifact_names[model_name])
        print(f"Saved {len(models)} model artifacts to {model_directory}")
        return

    cv_rows = []
    domain_rows = []
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        cv_rows.append({"model": model_name, **evaluate_cv(model_name, model, data, args.cv_folds, args.seed)})
        domain_rows.append({"model": model_name, **evaluate_domain_holdout(model, data, args.seed)})

    champion_name = "XGBoost" if "XGBoost" in models else "Random Forest"
    champion = models[champion_name]
    ablation_rows = run_ablation(champion, data, args.cv_folds, args.seed, ABLATION_GROUPS)
    cumulative_ablation_rows = run_ablation(
        champion, data, args.cv_folds, args.seed, CUMULATIVE_ABLATION_GROUPS
    )

    model_directory = Path(__file__).resolve().parents[1] / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths = {}
    if args.save_all_models:
        for model_name, model in models.items():
            fitted_model = clone(model).fit(data[FEATURE_NAMES], data["label"])
            artifact_name = {
                "XGBoost": "xgboost_phish.pkl",
                "Random Forest": "random_forest_phish.pkl",
                "SVM (RBF approximation)": "svm_rbf_approximation.pkl",
                "Decision Tree": "decision_tree_phish.pkl",
                "Logistic Regression": "logistic_regression_phish.pkl",
            }[model_name]
            artifact_path = model_directory / artifact_name
            joblib.dump(fitted_model, artifact_path)
            artifact_paths[model_name] = str(artifact_path)
    else:
        champion.fit(data[FEATURE_NAMES], data["label"])
        artifact_name = "xgboost_phish.pkl" if champion_name == "XGBoost" else "random_forest_phish.pkl"
        artifact_path = model_directory / artifact_name
        joblib.dump(champion, artifact_path)
        artifact_paths[champion_name] = str(artifact_path)

    metadata = {
        "dataset": str(args.dataset),
        "rows_used": len(data),
        "columns_used": ["URL", "Domain", "label"],
        "feature_names": FEATURE_NAMES,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "champion_saved_to": artifact_paths[champion_name],
        "artifact_paths": artifact_paths,
        "champion_model": champion_name,
        "xgboost_skipped": args.skip_xgboost,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(args.output_dir / "cross_validation_results.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(args.output_dir / "domain_disjoint_results.csv", index=False)
    pd.DataFrame(ablation_rows).to_json(args.output_dir / "ablation_results.json", orient="records", indent=2)
    pd.DataFrame(cumulative_ablation_rows).to_json(
        args.output_dir / "cumulative_ablation_results.json", orient="records", indent=2
    )
    if args.external_dataset:
        pd.DataFrame(evaluate_external(models, data, args.external_dataset, args.seed)).to_csv(
            args.output_dir / "external_results.csv", index=False
        )
    print(f"Completed {len(data):,} rows; champion: {champion_name}; artifacts: {len(artifact_paths)}")


if __name__ == "__main__":
    main()
