# PhiUSIIL URL-Only Experiments

This runner implements the main experiment defined in `PhiUSIIL_dataset_report.md`:

- same 12 raw-URL features for every model
- Logistic Regression, Decision Tree, Random Forest, scalable SVM with an RBF-kernel approximation, and XGBoost
- stratified cross-validation
- domain-disjoint holdout evaluation
- champion feature-group ablation
- full-data champion artifact for the FastAPI endpoint

The runner reads only `URL`, `Domain`, and `label` from the 56-column CSV. It does not use webpage-content columns, `FILENAME`, or the dataset's high-association derived fields.

## Complete run

From `website-react/backend`:

```powershell
python experiments/run_experiments.py `
  --dataset "C:\Users\91965\Downloads\PhiUSIIL_Phishing_URL_Dataset.csv" `
  --output-dir experiments/results `
  --save-all-models
```

Install XGBoost first if the command reports that it is missing:

```powershell
python -m pip install xgboost
```

## Smoke run without XGBoost

The current selected environment does not have XGBoost installed. To validate the pipeline locally with the other four models:

```powershell
python experiments/run_experiments.py `
  --dataset "C:\Users\91965\Downloads\PhiUSIIL_Phishing_URL_Dataset.csv" `
  --sample-size 4000 `
  --cv-folds 3 `
  --skip-xgboost `
  --output-dir experiments/smoke-results
```

## Outputs

- `cross_validation_results.csv`: mean accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- `domain_disjoint_results.csv`: the same metrics on an unseen-domain holdout
- `ablation_results.json`: champion results for structure, security, randomness, and all 12 features
- `cumulative_ablation_results.json`: progressive lexical-to-all-feature ablation results
- `run_metadata.json`: feature order, seed, rows used, and artifact location
- `../models/*.pkl`: full-data model artifacts when `--save-all-models` is enabled

The script extracts features from raw URLs using `backend/extract_features.py`, so the saved XGBoost model has the same feature order expected by the API.

Classic exact `SVC(kernel="rbf")` is not computationally practical for 235,795 rows. The SVM comparison therefore uses a 50-component `Nystroem(kernel="rbf")` map followed by `LinearSVC`, with a bounded sigmoid applied to decision scores for ranking metrics. This is an explicit scalable approximation of the RBF decision function, and it should be named as such in publication tables.

The complete run can take substantially longer than the smoke run because it extracts features for all 235,795 rows and trains five models across multiple folds. Treat the generated CSV and JSON files as experiment artifacts and record the command, environment versions, and dataset checksum with the paper.

## Completed full run

The full run has been completed successfully with 235,795 rows, five folds, and seed 42. XGBoost is the selected champion and was saved to `../models/xgboost_phish.pkl`. The result files are in `experiments/results/`.

The API selector requires the `--save-all-models` run to serve all five model choices. Without that flag, only the selected champion artifact is created and other API selections correctly return HTTP 503 rather than silently using XGBoost.
