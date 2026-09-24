# PhiUSIIL Phishing URL Dataset Report

**File analyzed:** `C:\Users\91965\Downloads\PhiUSIIL_Phishing_URL_Dataset.csv`  
**Analysis date:** 2026-08-22  
**Analysis scope:** Complete CSV scan using pandas with `low_memory=False`

## 1. Executive Summary

The dataset contains **235,795 URL records** and **56 columns**. It is a supervised binary classification dataset with the target column `label`:

| Class | Label value | Records | Share |
|---|---:|---:|---:|
| Legitimate | 0 | 100,945 | 42.81% |
| Phishing | 1 | 134,850 | 57.19% |
| **Total** |  | **235,795** | **100.00%** |

Important findings:

- There are **no missing values** in any of the 56 columns.
- There are **no exact duplicate rows**.
- There are **235,370 unique URLs**, meaning **425 records share a URL with another record**.
- Every repeated URL has a consistent label in this scan; however, repeated URLs must still be grouped before splitting to avoid train/test leakage.
- The dataset includes both **URL-only lexical features** and **webpage-content features**. It is therefore broader than the 12-feature, no-scraping architecture used by the PhishGuard browser prototype.
- Several variables have very strong univariate association with `label`, especially `URLSimilarityIndex` and page-content indicators. These should be examined carefully for collection or label leakage before publication.
- `FILENAME` is unique for every row and should normally be excluded from model training.

## 2. Dataset Shape and Storage

| Property | Observed value |
|---|---:|
| Rows | 235,795 |
| Columns | 56 |
| CSV file size | 56,854,345 bytes, approximately 54.2 MiB |
| Loaded pandas memory | 120.07 MiB |
| Numeric columns | 51 |
| Text/string columns | 5 |
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Unique URLs | 235,370 |
| Unique domains | 220,086 |
| Unique filenames | 235,795 |
| Unique TLD values | 695 |

The five string columns are `FILENAME`, `URL`, `Domain`, `TLD`, and `Title`. The remaining columns are numeric integer or floating-point variables. Most binary flags are represented as integer `0` and `1`.

## 3. Complete Column Dictionary

### 3.1 Identification and target fields

| Column | Observed type | Meaning and modeling guidance |
|---|---|---|
| `FILENAME` | text | Source/sample identifier. It is unique for all 235,795 records. Exclude from model features unless it has a documented, non-identifying purpose. |
| `URL` | text | Full URL string. Required for lexical parsing, normalization, duplicate grouping, and final demonstrations. Do not feed raw URL text into a model unless that is the declared experiment. |
| `Domain` | text | Extracted hostname/domain. Useful for grouping and leakage-aware splitting. Avoid using it as an unrestricted categorical predictor because domain memorization can inflate performance. |
| `TLD` | text | Top-level domain value, such as `com`, `org`, or `net`. It has 695 unique values and can encode geographic or collection-source bias. |
| `label` | integer | Binary target: `0` = legitimate and `1` = phishing. This is the supervised learning target and must not be included among input features. |

### 3.2 URL and lexical structure fields

| Column | Observed type | Meaning |
|---|---|---|
| `URLLength` | integer | Total URL character length. Observed range: 13 to 6,097; median: 27. |
| `DomainLength` | integer | Host/domain character length. Observed range: 4 to 110; median: 20. |
| `IsDomainIP` | integer flag | Whether the domain is represented as an IP address. 638 records have value `1`. |
| `URLSimilarityIndex` | float | Similarity score supplied by the dataset. Range: 0.155574 to 100; median and upper quartile: 100. This variable has the strongest observed absolute numeric correlation with `label` (approximately 0.860) and requires leakage investigation. |
| `CharContinuationRate` | float | Character continuation or repetition-related URL statistic. Range: 0 to 1; median: 1. |
| `TLDLegitimateProb` | float | Dataset-provided legitimacy probability associated with the TLD. Range: 0 to 0.5229071; median: approximately 0.079963. Treat as a derived feature, not an independent raw observation. |
| `URLCharProb` | float | Dataset-provided character probability score for the URL. Range: 0.001083 to 0.090824; median: approximately 0.057970. |
| `TLDLength` | integer | Number of characters in the TLD. Range: 2 to 13; median: 3. |
| `NoOfSubDomain` | integer | Number of subdomain components. Range: 0 to 10; median: 1. |
| `HasObfuscation` | integer flag | Whether URL obfuscation was detected. 0.206% of rows have value `1`. |
| `NoOfObfuscatedChar` | integer | Count of obfuscated characters. Range: 0 to 447; median: 0. |
| `ObfuscationRatio` | float | Ratio of obfuscated characters to URL characters. Range: 0 to 0.348; median: 0. |
| `NoOfLettersInURL` | integer | Number of alphabetic characters in the URL. Range: 0 to 5,191; median: 14. |
| `LetterRatioInURL` | float | Proportion of URL characters that are letters. Range: 0 to 0.926; median: 0.519. |
| `NoOfDegitsInURL` | integer | Number of digits in the URL. The source spelling is `Degits`. Range: 0 to 2,011; median: 0. |
| `DegitRatioInURL` | float | Proportion of URL characters that are digits. The source spelling is `Degit`. Range: 0 to 0.684; median: 0. |
| `NoOfEqualsInURL` | integer | Number of `=` characters. Range: 0 to 176; median: 0. |
| `NoOfQMarkInURL` | integer | Number of question marks. Range: 0 to 4; median: 0. |
| `NoOfAmpersandInURL` | integer | Number of ampersands. Range: 0 to 149; median: 0. |
| `NoOfOtherSpecialCharsInURL` | integer | Count of other special characters. Range: 0 to 499; median: 1. |
| `SpacialCharRatioInURL` | float | Ratio of special characters to URL characters. The source spelling is `Spacial`. Range: 0 to 0.397; median: 0.050. |
| `IsHTTPS` | integer flag | Whether the URL uses HTTPS. 184,539 records are HTTPS and 51,256 are not. |

### 3.3 Page-content and HTML behavior fields

These fields require webpage inspection or an upstream crawler. They are not compatible with a strict URL-only, no-network inference claim unless they are removed from the deployed feature vector.

| Column | Observed type | Meaning |
|---|---|---|
| `LineOfCode` | integer | Number of lines observed in page source. Range: 2 to 442,666; median: 429. |
| `LargestLineLength` | integer | Length of the largest source line. Range: 22 to 13,975,730; median: 1,090. |
| `HasTitle` | integer flag | Whether the page has a title element. 86.13% of rows are positive. |
| `Title` | text | Extracted page title. 197,874 unique values. The sample shows mojibake in some non-English title values, so encoding should be checked before text modeling. |
| `DomainTitleMatchScore` | float | Similarity between domain and page title. Range: 0 to 100; median: 0. |
| `URLTitleMatchScore` | float | Similarity between URL and page title. Range: 0 to 100; median: 100. |
| `HasFavicon` | integer flag | Whether a favicon was detected. Mean: 0.362. |
| `Robots` | integer flag | Whether robots metadata or a robots resource was detected. Mean: 0.267. |
| `IsResponsive` | integer flag | Whether responsive behavior was detected. Mean: 0.625. |
| `NoOfURLRedirect` | integer | Number of URL redirects. Range: 0 to 1; mean: 0.133. |
| `NoOfSelfRedirect` | integer | Number of self-redirects. Range: 0 to 1; mean: 0.040. |
| `HasDescription` | integer flag | Whether a page description was detected. Mean: 0.440. |
| `NoOfPopup` | integer | Number of popups. Range: 0 to 602; median: 0. |
| `NoOfiFrame` | integer | Number of iframes. Range: 0 to 1,602; median: 0. |
| `HasExternalFormSubmit` | integer flag | Whether a form submits to an external destination. Mean: 0.044. |
| `HasSocialNet` | integer flag | Whether social-network signals were detected. Mean: 0.457. |
| `HasSubmitButton` | integer flag | Whether a submit button was detected. Mean: 0.414. |
| `HasHiddenFields` | integer flag | Whether hidden form fields were detected. Mean: 0.378. |
| `HasPasswordField` | integer flag | Whether a password field was detected. Mean: 0.102. |
| `Bank` | integer flag | Whether banking-related content or terminology was detected. Mean: 0.127. |
| `Pay` | integer flag | Whether payment-related content or terminology was detected. Mean: 0.237. |
| `Crypto` | integer flag | Whether cryptocurrency-related content or terminology was detected. Mean: 0.023. |
| `HasCopyrightInfo` | integer flag | Whether copyright information was detected. Mean: 0.487. |
| `NoOfImage` | integer | Number of images. Range: 0 to 8,956; median: 8. |
| `NoOfCSS` | integer | Number of CSS resources or references. Range: 0 to 35,820; median: 2. |
| `NoOfJS` | integer | Number of JavaScript resources or references. Range: 0 to 6,957; median: 6. |
| `NoOfSelfRef` | integer | Number of same-domain references. Range: 0 to 27,397; median: 12. |
| `NoOfEmptyRef` | integer | Number of empty references. Range: 0 to 4,887; median: 0. |
| `NoOfExternalRef` | integer | Number of external references. Range: 0 to 27,516; median: 10. |

## 4. Label and Category Distributions

### 4.1 Protocol distribution

| `IsHTTPS` | Records | Legitimate share | Phishing share |
|---:|---:|---:|---:|
| 0 | 51,256 | 100.00% | 0.00% |
| 1 | 184,539 | 26.93% | 73.07% |

This is a very strong relationship. It may be a real property of the collection, but it can also reflect how examples were sourced or labeled. Do not claim HTTPS alone is a reliable phishing detector.

### 4.2 IP-domain distribution

| `IsDomainIP` | Records | Legitimate share | Phishing share |
|---:|---:|---:|---:|
| 0 | 235,157 | 42.66% | 57.34% |
| 1 | 638 | 100.00% | 0.00% |

The IP-domain flag is perfectly associated with label `0` in this file, which is counterintuitive for many phishing-threat assumptions. This should be investigated as a possible dataset construction artifact before using it as a security conclusion.

### 4.3 Most frequent TLD values

| TLD | Total | Legitimate | Phishing |
|---|---:|---:|---:|
| `com` | 112,554 | 43,769 | 68,785 |
| `org` | 18,793 | 2,269 | 16,524 |
| `net` | 7,097 | 3,099 | 3,998 |
| `app` | 6,508 | 6,368 | 140 |
| `uk` | 6,395 | 322 | 6,073 |
| `co` | 5,422 | 4,964 | 458 |
| `io` | 4,201 | 3,769 | 432 |
| `de` | 3,996 | 686 | 3,310 |
| `ru` | 3,875 | 2,983 | 892 |
| `au` | 2,979 | 373 | 2,606 |
| `dev` | 2,345 | 2,312 | 33 |
| `top` | 2,329 | 2,327 | 2 |
| `jp` | 2,219 | 137 | 2,082 |
| `it` | 1,887 | 218 | 1,669 |
| `edu` | 1,861 | 5 | 1,856 |

The TLD distribution is highly uneven. A random row split can allow a model to learn source or TLD patterns rather than general phishing structure.

## 5. URL and Row Duplication

- Exact full-row duplicates: **0**.
- Unique URLs: **235,370** out of **235,795** rows.
- Repeated URL records: **425 rows beyond the first occurrence**.
- All repeated URL groups observed in the scan had one unique label value.

Recommended handling:

1. Normalize URLs before deduplication: trim whitespace, normalize scheme handling, and apply a documented case policy.
2. Group by normalized URL or domain family before train/test splitting.
3. Report both row-level and group-level evaluation results.
4. Keep repeated observations in a separate audit table if they represent collection-time variation.

## 6. Numeric Feature Summary

The following values are the main descriptive statistics from the complete file. Percentiles are included to show skew and outliers.

| Feature | Min | 1% | Median | 99% | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `URLLength` | 13 | 17 | 27 | 144 | 6,097 | 34.57 |
| `DomainLength` | 4 | 8 | 20 | 57 | 110 | 21.47 |
| `NoOfSubDomain` | 0 | 0 | 1 | 3 | 10 | 1.16 |
| `NoOfObfuscatedChar` | 0 | 0 | 0 | 0 | 447 | 0.02 |
| `NoOfLettersInURL` | 0 | 4 | 14 | 99 | 5,191 | 19.43 |
| `NoOfDegitsInURL` | 0 | 0 | 0 | 27 | 2,011 | 1.88 |
| `NoOfEqualsInURL` | 0 | 0 | 0 | 2 | 176 | 0.06 |
| `NoOfQMarkInURL` | 0 | 0 | 0 | 1 | 4 | 0.03 |
| `NoOfAmpersandInURL` | 0 | 0 | 0 | 0 | 149 | 0.03 |
| `NoOfOtherSpecialCharsInURL` | 0 | 1 | 1 | 13 | 499 | 2.34 |
| `LineOfCode` | 2 | 2 | 429 | 10,388 | 442,666 | 1,141.90 |
| `LargestLineLength` | 22 | 30 | 1,090 | 129,412 | 13,975,730 | 12,789.53 |
| `NoOfPopup` | 0 | 0 | 0 | 4 | 602 | 0.22 |
| `NoOfiFrame` | 0 | 0 | 0 | 21 | 1,602 | 1.59 |
| `NoOfImage` | 0 | 0 | 8 | 243 | 8,956 | 26.08 |
| `NoOfCSS` | 0 | 0 | 2 | 51 | 35,820 | 6.33 |
| `NoOfJS` | 0 | 0 | 6 | 64 | 6,957 | 10.52 |
| `NoOfSelfRef` | 0 | 0 | 12 | 517 | 27,397 | 65.07 |
| `NoOfEmptyRef` | 0 | 0 | 0 | 31 | 4,887 | 2.38 |
| `NoOfExternalRef` | 0 | 0 | 10 | 423 | 27,516 | 49.26 |

The extreme maxima show heavy right skew. Tree models can tolerate this better than linear models, but transformations, clipping, or robust scaling should be evaluated for Logistic Regression and SVM.

## 7. Potential Leakage and Validity Risks

### 7.1 High association with the target

The strongest absolute Pearson correlations between numeric columns and `label` in this scan were:

| Feature | Absolute correlation |
|---|---:|
| `URLSimilarityIndex` | 0.860 |
| `HasSocialNet` | 0.784 |
| `HasCopyrightInfo` | 0.743 |
| `HasDescription` | 0.690 |
| `IsHTTPS` | 0.609 |
| `DomainTitleMatchScore` | 0.585 |
| `HasSubmitButton` | 0.578 |
| `IsResponsive` | 0.548 |
| `URLTitleMatchScore` | 0.539 |
| `SpacialCharRatioInURL` | 0.533 |
| `HasHiddenFields` | 0.508 |
| `HasFavicon` | 0.494 |
| `URLCharProb` | 0.469 |
| `CharContinuationRate` | 0.467 |
| `HasTitle` | 0.459 |

Correlation is not proof of leakage, but values this large require a feature-provenance review. In particular, any feature calculated using labels, source-specific rules, or post-collection information can make evaluation unrealistically easy.

### 7.2 Feature availability mismatch

The current PhishGuard backend is designed for URL-only inference under low latency. The dataset contains page features such as `LineOfCode`, `Title`, `HasPasswordField`, and reference counts. These cannot be computed without requesting and parsing page content.

For a valid URL-only experiment, use only features derived from the URL string, such as:

- `URLLength`
- `DomainLength`
- `IsDomainIP`
- `TLDLength`
- `NoOfSubDomain`
- `HasObfuscation`
- `NoOfObfuscatedChar`
- `ObfuscationRatio`
- `NoOfLettersInURL`
- `LetterRatioInURL`
- `NoOfDegitsInURL`
- `DegitRatioInURL`
- `NoOfEqualsInURL`
- `NoOfQMarkInURL`
- `NoOfAmpersandInURL`
- `NoOfOtherSpecialCharsInURL`
- `SpacialCharRatioInURL`
- `IsHTTPS`

The deployed prototype currently exposes a related 12-feature lexical contract with its own names. A training pipeline should explicitly map dataset columns to the backend contract rather than silently passing all 56 columns.

The canonical feature order is explicitly defined in `website-react/backend/extract_features.py` as `FEATURE_ORDER` and is imported by both the experiment runner and the FastAPI service. Model vectors are built from that list, not from implicit dictionary iteration. This makes training and inference order reproducible.

### 7.3 Source spelling and encoding

The dataset includes source spellings such as `NoOfDegitsInURL`, `DegitRatioInURL`, and `SpacialCharRatioInURL`. Preserve the original names in raw-data documentation, but use corrected internal aliases only through an explicit mapping table.

Some `Title` values display mojibake, for example sequences resembling `à¸...`. This suggests that at least some text was decoded with the wrong character encoding during collection or export. Treat `Title` as an optional field and validate its encoding before any NLP or title-based model.

The `suspicious_keywords` feature counts the number of distinct predefined keywords present in the URL. Repeated occurrences of one keyword count once; for example, three appearances of `login` still contribute one keyword. The current indicator function is rule-based and should be described as **rule-based risk indicators**, not as model feature attribution. Genuine model explanation, such as SHAP values for XGBoost, is a separate future extension.

## 8. Recommended Experimental Design

### 8.1 Data preparation

1. Preserve the raw CSV unchanged as the immutable source artifact.
2. Validate column names and numeric ranges.
3. Normalize URL strings for grouping, without overwriting the raw `URL` column.
4. Remove or quarantine `FILENAME` from model features.
5. Check repeated URLs, domains, and registrable-domain families.
6. Review provenance for derived variables such as `URLSimilarityIndex`, `TLDLegitimateProb`, and `URLCharProb`.
7. Decide whether the experiment is URL-only, page-content-based, or a controlled comparison of both.

### 8.2 Splitting strategy

Report at least four evaluation tiers:

| Tier | Split | Purpose |
|---|---|---|
| 1 | Stratified random split | Basic reproducibility baseline. |
| 2 | Stratified cross-validation | Variance and stability estimate. |
| 3 | Domain-disjoint split | Tests whether the model generalizes beyond seen domains. |
| 4 | Time- or source-disjoint external test | Tests distribution shift and deployment realism. |

For tiers 3 and 4, grouping by registrable domain and source is more informative than splitting individual rows. Report the number of unique domains in every split.

### 8.3 Metrics

Report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC, especially because the classes are not perfectly balanced
- Confusion matrix
- Inference latency per URL
- Feature extraction latency separately from model inference latency
- Calibration or reliability metrics for confidence scores

Use confidence intervals or repeated splits where practical. A single random split is not sufficient evidence of generalization.

## 9. Relationship to the PhishGuard Application

The application has two deployment modes:

```text
React frontend
    |
    | optional POST /api/scan
    v
FastAPI backend
    |
    +-- URL feature extraction
    +-- trained XGBoost model, when installed
    +-- deterministic heuristic proxy, otherwise
```

The CSV can support model training, but the current backend does not automatically load arbitrary CSV columns. Before placing a serialized model at `website-react/backend/models/xgboost_phish.pkl`, the model must be trained with exactly the same ordered feature vector returned by `extract_features.py`.

The current API response includes:

- submitted URL
- selected model name
- risk score
- phishing or legitimate verdict
- extracted feature dictionary
- explanation messages
- `trained-model` or `heuristic-proxy` inference mode

## 10. Reproducibility Checklist

- [ ] Record the dataset filename, checksum, and acquisition date.
- [ ] Record the Python, pandas, scikit-learn, and XGBoost versions.
- [ ] Keep raw and processed datasets separate.
- [ ] Publish the exact column-selection and renaming map.
- [ ] Exclude `FILENAME` and `label` from predictors.
- [ ] Document whether page-content features were available at prediction time.
- [ ] Group duplicate URLs and domains before evaluation.
- [ ] Report all four evaluation tiers.
- [ ] Store random seeds and split manifests.
- [ ] Report feature extraction and inference latency independently.
- [ ] Audit high-association variables for target leakage.
- [ ] Validate title encoding before using `Title`.
- [ ] Do not describe the dataset as balanced: the observed phishing share is 57.19%.

## 11. Bottom Line

This is a large, mostly complete, feature-rich binary classification dataset suitable for a serious phishing URL study. Its main strengths are scale, broad URL and page-derived feature coverage, and a usable binary target. Its main risks are source bias, strong target associations, page-content availability mismatch, duplicate domains, and possible encoding problems in title text.

For the current lightweight PhishGuard architecture, the most defensible approach is to train a dedicated URL-only model using a documented lexical subset, exclude identifiers and page-content fields, use domain-aware splits, and clearly label the browser/API heuristic as a proxy until a matching trained model is installed.

## 12. Executable Experiment Implementation

The executable runner is located at `website-react/backend/experiments/run_experiments.py`. It implements the main experiment design in this report:

- extracts the same 12 raw-URL features used by the FastAPI backend;
- compares Logistic Regression, Decision Tree, Random Forest, SVM (RBF), and XGBoost;
- calculates accuracy, precision, recall, F1, ROC-AUC, and PR-AUC;
- evaluates stratified cross-validation and a domain-disjoint holdout;
- runs feature-group ablation for structure, security, randomness, and all 12 features;
- saves metadata and a full-data champion model for API integration.

The runner was smoke-tested against the real CSV with 4,000 stratified rows and three folds. XGBoost was skipped because it was not installed in the selected Python 3.14 environment. The four-model smoke results were:

| Model | Cross-validation F1 | Domain-disjoint F1 |
|---|---:|---:|
| Logistic Regression | 0.9940 | 0.9948 |
| Decision Tree | 0.9950 | 0.9974 |
| Random Forest | 0.9955 | 0.9934 |
| SVM (RBF) | 0.9948 | 0.9961 |

These values are pipeline validation evidence, not final publication results. Run the complete experiment with XGBoost installed before using metrics in the paper. The generated smoke outputs are stored under `website-react/backend/experiments/smoke-results/`.

## 13. Full URL-Only Experiment Results

The complete experiment was subsequently executed on **all 235,795 rows** using the 12 URL-only features, seed `42`, and five stratified folds. The five model paths completed successfully. The SVM row uses the documented scalable RBF approximation (`Nystroem` plus `LinearSVC`), not an exact quadratic-cost RBF SVC.

The runner preserves fold variability: every cross-validation and ablation metric includes a companion `_std` field calculated with sample standard deviation (`ddof=1`). The cross-validation and ablation tables below report mean $\pm$ standard deviation.

### 13.1 Stratified five-fold cross-validation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9933 ± 0.0004 | 0.9890 ± 0.0006 | 0.9994 ± 0.0003 | 0.9942 ± 0.0004 | 0.9961 ± 0.0003 | 0.9943 ± 0.0007 |
| Decision Tree | 0.9951 ± 0.0004 | 0.9923 ± 0.0005 | 0.9992 ± 0.0002 | 0.9957 ± 0.0003 | 0.9968 ± 0.0004 | 0.9953 ± 0.0005 |
| Random Forest | 0.9949 ± 0.0002 | 0.9933 ± 0.0004 | 0.9979 ± 0.0003 | 0.9956 ± 0.0002 | 0.9965 ± 0.0004 | 0.9947 ± 0.0006 |
| SVM (RBF approximation) | 0.9784 ± 0.0073 | 0.9919 ± 0.0014 | 0.9701 ± 0.0125 | 0.9809 ± 0.0066 | 0.9955 ± 0.0005 | 0.9937 ± 0.0010 |
| **XGBoost** | **0.9956 ± 0.0003** | **0.9932 ± 0.0005** | **0.9993 ± 0.0002** | **0.9962 ± 0.0003** | **0.9979 ± 0.0002** | **0.9972 ± 0.0003** |

XGBoost is the cross-validation champion by F1, ROC-AUC, and PR-AUC. Its very high recall means it identifies nearly all phishing examples in this dataset, while precision remains above 0.99. These results are strong but should not be interpreted as deployment accuracy without independent external validation.

### 13.2 Domain-disjoint holdout

The domain-disjoint split holds out domain groups rather than individual rows, reducing direct domain memorization between training and test partitions.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9937 | 0.9897 | 0.9995 | 0.9946 | 0.9960 | 0.9936 |
| Decision Tree | 0.9957 | 0.9932 | 0.9994 | 0.9963 | 0.9972 | 0.9959 |
| Random Forest | 0.9953 | 0.9944 | 0.9975 | 0.9959 | 0.9969 | 0.9953 |
| SVM (RBF approximation) | 0.9753 | 0.9921 | 0.9648 | 0.9783 | 0.9926 | 0.9926 |
| **XGBoost** | **0.9960** | **0.9940** | **0.9991** | **0.9966** | **0.9984** | **0.9980** |

The domain-disjoint results are not lower than the random cross-validation results. This is unusual for a dataset where domain memorization is a concern and should be investigated rather than presented as proof of universal generalization. The split is grouped by the dataset's `Domain` field, not a separately resolved registrable-domain family, and the source may contain strong collection artifacts.

### 13.3 Full-data ablation study

The XGBoost champion was evaluated with feature groups under the same five-fold cross-validation procedure:

| Feature set | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| All 12 features | 0.9956 ± 0.0003 | 0.9962 ± 0.0003 | 0.9979 ± 0.0002 | 0.9972 ± 0.0003 |
| Structure only | 0.9924 ± 0.0007 | 0.9934 ± 0.0006 | 0.9970 ± 0.0001 | 0.9959 ± 0.0003 |
| Security only | 0.8050 ± 0.0010 | 0.8537 ± 0.0006 | 0.7746 ± 0.0012 | 0.7477 ± 0.0010 |
| Randomness only | 0.7849 ± 0.0014 | 0.8337 ± 0.0011 | 0.8056 ± 0.0019 | 0.7878 ± 0.0031 |

### 13.4 Cumulative ablation

The cumulative sequence makes the contribution of feature families easier to interpret:

| Feature set | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Lexical | 0.8060 ± 0.0010 | 0.8481 ± 0.0007 | 0.8239 ± 0.0009 | 0.7999 ± 0.0013 |
| Lexical + structural | 0.8617 ± 0.0012 | 0.8909 ± 0.0009 | 0.8836 ± 0.0010 | 0.8651 ± 0.0014 |
| + Domain length | 0.9930 ± 0.0011 | 0.9939 ± 0.0010 | 0.9976 ± 0.0001 | 0.9967 ± 0.0002 |
| + Security | 0.9957 ± 0.0003 | 0.9962 ± 0.0003 | 0.9979 ± 0.0001 | 0.9973 ± 0.0003 |
| + Entropy | 0.9956 ± 0.0003 | 0.9962 ± 0.0003 | 0.9979 ± 0.0002 | 0.9973 ± 0.0003 |

The largest cumulative gain occurs when domain length is added to the lexical and structural features. Security features then provide a smaller improvement, while entropy contributes little additional gain in this dataset once the other feature families are present.

The structure-only group performs strongly, while security-only and randomness-only groups are insufficient on their own. Combining the feature groups improves the overall result. The strongest conclusion supported by this ablation is that multiple lightweight URL signals are complementary; it does not establish that each individual feature is causally necessary.

### 13.5 Generated artifacts

The final machine-readable results are stored in:

- `website-react/backend/experiments/results/cross_validation_results.csv`
- `website-react/backend/experiments/results/domain_disjoint_results.csv`
- `website-react/backend/experiments/results/ablation_results.json`
- `website-react/backend/experiments/results/cumulative_ablation_results.json`
- `website-react/backend/experiments/results/run_metadata.json`
- `website-react/backend/models/xgboost_phish.pkl`

The model artifact was trained using the exact feature order listed in `run_metadata.json` and is compatible with the FastAPI feature extraction contract. External dataset testing remains outstanding because no second labeled dataset is currently available in the workspace.

No external labeled dataset was found in `C:\Users\91965\Downloads` during the current validation; only `PhiUSIIL_Phishing_URL_Dataset.csv` was present. The optional `--external-dataset` evaluator is implemented, but external metrics must wait for a second dataset with a compatible `URL` and `label` schema.

## 14. Implementation Consistency Status

The application and experiment implementation are now aligned:

- XGBoost uses `n_estimators=150`, `max_depth=6`, and `learning_rate=0.1` in both the runner documentation and frontend model description.
- Random Forest uses `n_estimators=150` with balanced class weights.
- The SVM is named `SVM (RBF approximation)` everywhere and uses a 50-component Nystroem RBF map followed by `LinearSVC`.
- The API maps each selector to its own artifact: XGBoost, Random Forest, SVM approximation, Decision Tree, and Logistic Regression.
- If a requested non-XGBoost artifact is absent, the API returns HTTP 503 rather than running a different model.
- When the FastAPI URL is not configured, the frontend labels its result `Demo heuristic - browser fallback`; it does not represent the heuristic as a trained ML model.
- Cumulative ablation and optional external CSV evaluation are implemented in `website-react/backend/experiments/run_experiments.py`.

All five full-data model artifacts were generated and each selector was verified through `POST /api/scan` with `inference_mode: trained-model`. External evaluation still requires a second labeled dataset and has not been claimed as completed.

## 15. URL-Only Leakage Audit

The full CSV was audited using `website-react/backend/experiments/leakage_audit.py`. The audit uses the same `FEATURE_ORDER` and `extract_features()` implementation as training and API inference.

| Audit check | Result |
|---|---:|
| Rows audited | 235,795 |
| Missing values in URL, Domain, or label | 0 |
| Exact duplicate rows among used columns | 425 |
| Repeated normalized URL groups | 425 |
| Mixed-label normalized URL groups | 0 |
| Repeated domain groups | 5,526 |
| Mixed-label domain groups | 54 |
| Mixed-label domain rate | 0.0245% |

The `425` repeated URL groups are label-consistent, so the audit found no direct contradiction where the same normalized URL has both labels. Nevertheless, repeated URLs should remain in one split. Domain repetition is much more common, and the `54` mixed-label domains demonstrate why domain-aware grouping is important for honest generalization estimates.

### 15.1 URL-only target associations

The audit found these Pearson correlations with the target label among the 12 deployed features:

| Feature | Correlation with label |
|---|---:|
| `is_https` | 0.6129 |
| `num_slashes` | -0.4822 |
| `url_entropy` | -0.3970 |
| `domain_length` | -0.2832 |
| `url_length` | -0.2282 |
| `num_hyphens` | -0.2028 |
| `num_digits` | -0.1784 |
| `suspicious_keywords` | -0.1676 |
| `special_chars` | -0.1493 |
| `num_dots` | -0.1196 |
| `has_ip` | -0.0602 |
| `num_subdomains` | -0.0060 |

These associations are not proof of leakage. They show that the URL-only feature set contains strong dataset-specific signal, especially HTTPS and URL structure. The unusually high model performance should therefore be reported with domain-disjoint results and, preferably, an independent external dataset. The audit artifact is stored at `website-react/backend/experiments/results/leakage_audit.json`.

### 15.2 Audit conclusion

The audit supports the implementation claim that training and inference use the same 12 URL-only definitions and that the experiment excludes webpage-content columns and `FILENAME`. It does not prove that the dataset is free from source or labeling bias. External validation remains the required next check before making broad real-world generalization claims.
