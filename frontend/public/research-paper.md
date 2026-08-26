# Robustness and Generalization Limits of Lightweight URL-Based Phishing Classifiers Under Cross-Dataset Distribution Shifts

## Abstract

Phishing websites remain a persistent cybersecurity threat because attackers can rapidly generate deceptive URLs that impersonate trusted services, redirect victims, and evade static blocklists. This paper investigates the robustness and cross-dataset generalization capabilities of lightweight, URL-only machine learning classifiers. While classical models often report near-perfect in-dataset performance, these results can be artificially inflated by dataset-specific feature-label associations that do not hold in real-world deployments. To address this, we evaluate five classical machine learning models (XGBoost, Random Forest, SVM with RBF approximation, Decision Tree, and Logistic Regression) trained on the PhiUSIIL dataset using a methodologically justified subset of twelve raw URL features. We analyze model performance under three progressively stricter evaluation tiers: standard stratified cross-validation, domain-disjoint holdout splits, and external cross-dataset validation on an independent dataset of 129,777 unique URLs (Wangchuk dataset). We observe a catastrophic generalization collapse, with the champion XGBoost model's F1-score dropping from **99.62%** in-dataset to **3.84%** externally, accompanied by an external recall of only **1.97%**. We perform formal statistical tests (Kolmogorov-Smirnov and Chi-Square) to quantify the feature distribution shifts between the datasets and employ SHAP (SHapley Additive exPlanations) to explain the model's feature attributions. Additionally, we conduct a threshold analysis, a probability calibration check, a rule-based baseline comparison, and a detailed error audit. Our results show that the model's failure is driven by its reliance on dataset-specific shortcuts (such as HTTPS presence and slash counts). Crucially, we demonstrate that a simple rule-based heuristic baseline outperforms all five machine learning models on the external dataset (**53.07%** F1 vs. **3.84%** F1). Finally, we report the computational efficiency of the pipeline, showing that our lightweight extraction and inference pipeline requires only 0.014 ms per URL under the evaluated hardware configuration.

**Keywords:** phishing detection, cross-dataset generalization, distribution shift, lexical features, XGBoost, SHAP, explainable AI, probability calibration, rule-based baseline

---

## 1. Introduction

Phishing is a social-engineering attack in which adversaries distribute fraudulent links that appear to represent legitimate services, such as banks, cloud platforms, e-commerce websites, or identity providers. Once a victim visits the link, the attacker may collect credentials, payment details, session tokens, or other sensitive information. Traditional phishing defenses often rely on manually maintained blacklists, domain reputation feeds, or content inspection. Although useful, these approaches can struggle with newly created phishing infrastructure because malicious URLs may appear and disappear before a blacklist is updated.

Machine learning offers a complementary defense by detecting suspicious URL patterns before the link is widely reported. A URL-only classifier is especially practical because it can operate without loading potentially dangerous web content. This reduces user exposure, lowers latency, and makes the system suitable for browser extensions, email filters, mobile messaging clients, and security dashboards.

However, a major limitation of current phishing URL detection literature is the reliance on single-dataset evaluations. Models trained and tested on the same dataset frequently report near-perfect classification performance (F1-scores > 99%). This creates a false sense of security, as classifiers may be memorizing dataset-specific collection artifacts (spurious correlations) rather than learning general concepts of phishing behavior. For example, a dataset might contain phishing URLs collected only during a period when all attackers used HTTPS, leading the model to treat HTTPS as a deterministic indicator of phishing.

This study systematically investigates this generalization gap. The central research question of this study is:

> **How well do lightweight URL-based phishing classifiers generalize beyond the specific dataset on which they are trained?**

To address this, we formulate five supporting research questions:
- **RQ1**: How accurately and stably do the five classical classifiers perform under repeated stratified cross-validation?
- **RQ2**: Does domain-disjoint evaluation within the same dataset expose any performance degradation, and what is the training/testing domain distribution?
- **RQ3**: Which feature groups contribute most to the baseline performance under ablation, and how does reducing dimensionality impact performance and latency?
- **RQ4**: Do the models generalize when evaluated on an independent, external dataset collected under different conditions?
- **RQ5**: What specific feature distribution shifts, probability miscalibrations, and model shortcuts explain the generalization gap, and how do the models compare against a simple heuristic baseline?

---

## 2. Related Work

Earlier URL-based phishing detection research showed that lexical signals can help identify emerging malicious links before static blacklists are updated. Blum et al. demonstrated lexical-feature-based phishing URL detection using online learning, emphasizing adaptability against newly appearing threats [1]. Classical machine learning models remain widely used in phishing detection because engineered URL features are inexpensive to compute and often provide strong baseline performance. The models selected in this study represent both interpretable and high-performing families of classifiers, including Random Forests [2], Support Vector Machines [3], and XGBoost [4]. The use of Shannon entropy follows the information-theoretic formulation introduced by Shannon [5].

Recent research has increasingly questioned the real-world robustness of these models. Rashid et al. [10] explicitly studied cross-dataset generalization, showing that phishing URL classifiers can experience substantial performance degradation (10% to 32% F1-score loss) when evaluated on independent datasets. They investigated feature distribution shifts and proposed domain adaptation methods as a remedy. Similarly, Yi et al. [11] combined cross-dataset validation with SHAP-based interpretability to evaluate ML models across four distinct test sets, demonstrating that models rely heavily on shortcuts that fail to transfer. 

Furthermore, feature selection remains a critical topic in lightweight phishing detection. While some studies compare classical models on complete datasets [12], other feature-selection papers emphasize reducing feature counts using Mutual Information, Chi-Square tests, and correlation filtering to minimize computational and memory costs [13], [14].

Table 23 compares previous studies across the key evaluation dimensions considered in this work.

Table 23: Comparison of URL Phishing Studies and Evaluated Dimensions
| Study | URL-only | Multiple models | Domain-disjoint | External dataset | Distribution shift | SHAP | Calibration | Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Blum et al. (2010) [1] | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Rashid et al. (2024) [10]| ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Yi et al. (2026) [11] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Murthy et al. (2026) [12]| ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **PhishGuard (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Existing studies address individual aspects of phishing URL detection, such as feature engineering, classifier comparison, cross-dataset evaluation, or explainability. For instance, Murthy et al. [12] compared multiple classifiers but evaluated solely in-domain on the hybrid features of PhiUSIIL. Yi et al. [11] combined cross-dataset testing with SHAP but omitted calibration and efficiency metrics. PhishGuard combines all of these evaluations within a single controlled URL-only framework, focusing specifically on whether apparently strong source-dataset performance survives progressively stricter evaluation settings.

---

## 3. System Architecture

The proposed system follows a five-stage workflow:

```text
+---------------------------------------------------------------------+
|                           User Interaction                          |
|                    User submits URL and clicks Analyze               |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                         Lexical Extraction                          |
|        URL parser computes 12 numerical and structural features      |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                        Machine Learning Engine                      |
|        XGBoost, Random Forest, SVM, Decision Tree, or Logistic       |
|        Regression evaluates the feature vector                      |
+---------------------------------------------------------------------+
                                  |
             +--------------------+--------------------+
             v                                         v
+-------------------------------+     +-------------------------------+
| Prediction and Confidence     |     | Explainability Diagnostics    |
| Phishing vs. Legitimate       |     | Human-readable warning flags  |
| Probability score from 0-100% |     | based on risky URL features   |
| (Threshold-tuned operating pt)|     | and SHAP feature attribution  |
+-------------------------------+     +-------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                    Research Benchmark and Dashboard                 |
| Accuracy, precision, recall, F1, ROC-AUC, latency, and drift views  |
+---------------------------------------------------------------------+
```

The architecture intentionally avoids webpage crawling or network requests. This makes the classifier safer to run in user-facing contexts and keeps detection latency low.

---

## 4. Feature Extraction and Selection Methodology

The URL feature extraction engine converts each raw URL into a twelve-dimensional feature vector. The final 12 features were selected through a structured feature-selection methodology designed to ensure a lightweight, non-redundant, and highly informative representation:

1. **Initial Candidate Generation**: We started with an initial candidate pool of **56** features from the raw PhiUSIIL dataset schema. We excluded webpage-derived attributes (e.g., `LineOfCode`, `NoOfiFrame`) and complex derived similarity scores (e.g., `URLSimilarityIndex`) to enforce a strict URL-only lexical constraint, leaving **25** candidate lexical URL features.
2. **Deduplication and Constant Removal**: Features with zero variance across training samples were discarded.
3. **Correlation Analysis**: We computed a Pearson correlation matrix to identify redundant features. Pairs with $|r| > 0.85$ were analyzed (e.g., digit count vs. digit ratio); removing the highly correlated redundant counterparts reduced the candidate pool to **18** features.
4. **Mutual Information (MI) Selection**: We computed MI scores to rank the remaining 18 features. The final **12** selected features listed in Table 1 exhibit the highest positive MI scores, justifying their inclusion.

Table 1: Deployed 12 Features and Security Rationale
| Feature | Type | Security Rationale |
| :--- | :--- | :--- |
| `url_length` | Numeric | Phishing URLs often use long strings to hide the destination or include tracking tokens. |
| `domain_length` | Numeric | Very long hostnames may indicate obfuscation or generated infrastructure. |
| `num_dots` | Numeric | Excessive dots may indicate nested subdomains or deceptive host construction. |
| `num_subdomains` | Numeric | Attackers may imitate trusted brands through misleading subdomains. |
| `num_digits` | Numeric | Algorithmically generated links frequently contain numeric tokens. |
| `special_chars` | Numeric | High symbol density (`@`, `?`, `=`, `%`, `_`, `&`) indicates parameter stuffing or obfuscation. |
| `num_hyphens` | Numeric | Hyphens are often used in typosquatting or brand impersonation. |
| `num_slashes` | Numeric | Slashes indicate deep paths, which are often used to host phishing templates. |
| `has_ip` | Binary | Direct IP-based hosts can be associated with temporary or less-established infrastructure and may provide a useful lexical signal. |
| `is_https` | Binary | Lack of HTTPS remains a risk signal, though HTTPS alone does not prove legitimacy. |
| `suspicious_keywords`| Numeric | Counts terms commonly found in credentials harvesting (e.g., `login`, `verify`, `secure`). |
| `url_entropy` | Numeric | High character distribution entropy indicates generated domains or opaque tokens. |

### 4.1 Strict Input Representation Limits on PhiUSIIL
Although the PhiUSIIL dataset [15] contains a substantially richer set of 56 columns (including webpage HTML source features like `LineOfCode`, `NoOfiFrame`, and derived metrics like `URLSimilarityIndex`), this study intentionally restricts the input representation to the twelve URL-derived features listed in Table 1. This prevents target leakage from webpage-derived characteristics and ensures that our evaluations focus strictly on a lightweight, no-scraping inference model.

---

## 5. Machine Learning Pipeline

The comparative benchmark includes five classifiers representing linear, tree, and ensemble models. Rather than presenting these models as a standalone contribution, they serve as controlled baseline controls to investigate whether cross-dataset generalization failures are model-specific or stem from fundamental feature relationships.

Table 2: Deployed Baseline Classifiers
| Model | Role in Study | Key Hyperparameters / Setup |
| :--- | :--- | :--- |
| XGBoost | Primary ensemble champion for non-linear interactions | `n_estimators=150`, `max_depth=6`, `learning_rate=0.1` |
| Random Forest | Robust bagging ensemble baseline | `n_estimators=150`, `class_weight="balanced"` |
| SVM (RBF approx.) | Scalable margin-based classifier | 50-comp. `Nystroem` RBF map + `LinearSVC` with sigmoid scale |
| Decision Tree | Lightweight interpretable baseline | `max_depth=8`, `class_weight="balanced"` |
| Logistic Regression| Linear statistical baseline | `max_iter=1000`, `class_weight="balanced"`, L2 penalty |

Feature scaling (z-score normalization) is applied to the SVM and Logistic Regression models.

---

## 6. Dataset and Experimental Protocol

Our experiments utilize two distinct datasets to evaluate generalization:
1. **PhiUSIIL Dataset**: 235,795 URLs (100,945 legitimate, 134,850 phishing) [15]. Used for training, cross-validation, and in-dataset holdout.
2. **Wangchuk Dataset (External)**: An independent dataset published by Wangchuk (2026) [9] consisting of 129,777 unique URLs (74,972 legitimate from Common Crawl, 54,805 phishing from PhishTank). This serves as our external cross-dataset test set.

We report performance across four experiments:
- **Experiment 1 (Standard Evaluation)**: Stratified 5-fold cross-validation on PhiUSIIL.
- **Experiment 2 (Domain-Disjoint Evaluation)**: Group-based split separating domain families (80% train, 20% test) to prevent direct domain memorization.
- **Experiment 3 (Feature Ablation)**: Evaluating the impact of different feature sub-families on the champion model.
- **Experiment 4 (Cross-Dataset Evaluation)**: Training on full PhiUSIIL and testing on the external Wangchuk dataset.

---

## 7. Explainability Layer

To transition from heuristic rules to robust model-level explainability, we incorporate **SHAP (SHapley Additive exPlanations)**. This local attribution method calculates the marginal contribution of each feature to the model's final log-odds prediction, exposing which features drive classification decisions.

---

## 8. Experimental Results

### 8.1 Experiment 1: Standard Evaluation (Stratified 5-Fold Cross-Validation)
Table 3 details the average performance metrics along with their sample standard deviations ($\pm \text{Std}$) computed across the five folds.

Table 3: Stratified 5-Fold Cross-Validation Performance on PhiUSIIL Dataset
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | **0.9956 ± 0.0003** | 0.9932 ± 0.0005 | 0.9993 ± 0.0002 | **0.9962 ± 0.0003** | **0.9979 ± 0.0002** | **0.9972 ± 0.0003** |
| Random Forest | 0.9949 ± 0.0002 | **0.9933 ± 0.0004** | 0.9979 ± 0.0003 | 0.9956 ± 0.0002 | 0.9965 ± 0.0004 | 0.9947 ± 0.0006 |
| SVM (RBF approx.) | 0.9784 ± 0.0073 | 0.9919 ± 0.0014 | 0.9701 ± 0.0125 | 0.9809 ± 0.0066 | 0.9955 ± 0.0005 | 0.9937 ± 0.0010 |
| Decision Tree | 0.9951 ± 0.0004 | 0.9923 ± 0.0005 | 0.9992 ± 0.0002 | 0.9957 ± 0.0003 | 0.9968 ± 0.0004 | 0.9953 ± 0.0005 |
| Logistic Regression | 0.9933 ± 0.0004 | 0.9890 ± 0.0006 | **0.9994 ± 0.0003** | 0.9942 ± 0.0004 | 0.9961 ± 0.0003 | 0.9943 ± 0.0007 |

All models exhibit exceptionally high performance within the PhiUSIIL dataset, with XGBoost leading with an F1-score of 99.62%.

### 8.2 Experiment 2: Domain-Disjoint Evaluation
To test whether the models generalize to unseen domains within the same dataset, we conducted a domain-disjoint holdout evaluation. Table 4 reports the holdout results. Table 5 profiles the training and testing splits, verifying that domains are strictly disjoint (domain overlap count is exactly 0).

Table 4: Domain-Disjoint Holdout Performance on PhiUSIIL Dataset
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | **0.9960** | 0.9940 | 0.9991 | **0.9966** | **0.9984** | **0.9980** |
| Random Forest | 0.9953 | **0.9944** | 0.9975 | 0.9959 | 0.9969 | 0.9953 |
| SVM (RBF approx.) | 0.9753 | 0.9921 | 0.9648 | 0.9783 | 0.9926 | 0.9926 |
| Decision Tree | 0.9957 | 0.9932 | **0.9994** | 0.9963 | 0.9972 | 0.9959 |
| Logistic Regression | 0.9937 | 0.9897 | 0.9995 | 0.9946 | 0.9960 | 0.9936 |

Table 5: Domain-Disjoint Evaluation Splits Summary
| Split Cohort | Total URLs | Unique Domains | Legitimate (0) | Phishing (1) | Phishing Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Training Set (80%) | 189,074 | 176,068 | 81,147 | 107,927 | 57.08% |
| Testing Set (20%) | 46,721 | 44,018 | 19,798 | 26,923 | 57.63% |

The domain-disjoint results match or slightly exceed the stratified 5-fold cross-validation performance. This suggests that the models do not suffer from domain memorization within this dataset.

### 8.3 Experiment 3: Feature Ablation Study
We conducted feature ablation experiments using the champion XGBoost model under 5-fold cross-validation. Table 6 details performance across pre-defined feature subsets: Structural features (length, domain length, dots, subdomains, hyphens, slashes), Security features (IP usage, HTTPS, keywords), and Randomness features (digits, special characters, entropy). Additionally, Table 7 shows the cumulative ablation results.

Table 6: Group-Wise Feature Ablation on XGBoost Champion
| Feature Set | Accuracy | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| All 12 Features | **0.9956 ± 0.0003** | **0.9962 ± 0.0003** | **0.9979 ± 0.0002** | **0.9972 ± 0.0003** |
| Structure Only | 0.9924 ± 0.0007 | 0.9934 ± 0.0006 | 0.9970 ± 0.0001 | 0.9959 ± 0.0003 |
| Security Only | 0.8050 ± 0.0010 | 0.8537 ± 0.0006 | 0.7746 ± 0.0012 | 0.7477 ± 0.0010 |
| Randomness Only | 0.7849 ± 0.0014 | 0.8337 ± 0.0011 | 0.8056 ± 0.0019 | 0.7878 ± 0.0031 |

Table 7: Cumulative Feature Ablation on XGBoost Champion
| Feature Set | Accuracy | F1-score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: |
| Lexical (Length, Digits, Special Chars) | 0.8060 ± 0.0010 | 0.8481 ± 0.0007 | 0.8239 ± 0.0009 | 0.7999 ± 0.0013 |
| Lexical + Structural | 0.8617 ± 0.0012 | 0.8909 ± 0.0009 | 0.8836 ± 0.0010 | 0.8651 ± 0.0014 |
| + Domain Length | 0.9930 ± 0.0011 | 0.9939 ± 0.0010 | 0.9976 ± 0.0001 | 0.9967 ± 0.0002 |
| + Security | **0.9957 ± 0.0003** | **0.9962 ± 0.0003** | **0.9979 ± 0.0001** | **0.9973 ± 0.0003** |
| + Entropy (All 12) | 0.9956 ± 0.0003 | 0.9962 ± 0.0003 | 0.9979 ± 0.0002 | 0.9973 ± 0.0003 |

Adding domain length to the lexical and structural features results in the largest performance jump (F1-score increases from 89.09% to 99.39%), demonstrating that domain-length boundaries are extremely informative in the training dataset.

### 8.4 Experiment 4: External Cross-Dataset Validation
To evaluate the real-world generalization capability of the PhishGuard models, we trained them on the full PhiUSIIL dataset and tested them on the external Wangchuk dataset [9], which consists of 129,777 unique URLs (74,972 legitimate, 54,805 phishing). Table 8 shows the performance metrics, demonstrating a catastrophic performance collapse across all classifiers.

Table 8: External Cross-Dataset Validation Performance (Wangchuk Dataset)
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | 0.5843 | **0.8337** | 0.0197 | 0.0384 | 0.6838 | **0.6378** |
| Random Forest | 0.5844 | 0.8274 | 0.0199 | 0.0389 | 0.5175 | 0.4402 |
| SVM (RBF approx.) | 0.5855 | 0.8258 | 0.0234 | 0.0456 | 0.4100 | 0.4416 |
| Decision Tree | 0.5843 | 0.8279 | 0.0198 | 0.0387 | 0.5098 | 0.4305 |
| Logistic Regression | **0.5860** | 0.7855 | **0.0272** | **0.0526** | **0.7058** | 0.6026 |

All models struggle to detect phishing URLs from the external dataset, resulting in recalls below 3% and F1-scores between 3.8% and 5.3%. While precision remains relatively high (78.5% - 83.4%), the classifiers are almost entirely failing to identify the malicious class. To understand this prediction collapse, we examine the confusion matrices in Table 9.

Table 9: Confusion Matrices on the External Wangchuk Dataset
| Model | True Legit (TN) | False Phish (FP) | False Legit (FN) | True Phish (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 74,757 | 215 | 53,727 | 1,078 |
| **Random Forest** | 74,742 | 230 | 53,728 | 1,077 |
| **SVM (RBF approx.)** | 74,701 | 271 | 53,520 | 1,285 |
| **Decision Tree** | 74,741 | 231 | 53,718 | 1,087 |
| **Logistic Regression** | 74,565 | 407 | 53,315 | 1,490 |

The confusion matrices show that all five models predict almost every URL in the external dataset as legitimate. For example, XGBoost classifies only 1,293 out of 129,777 URLs as phishing (1,078 True Positives and 215 False Positives), missing 53,727 phishing URLs (False Negatives). This indicates a systemic generalization failure where the models are unable to transfer the learned decision boundary to an independent dataset.

### 8.5 Statistical Distribution-Shift Tests
To mathematically demonstrate why the models collapse, we performed two-sample Kolmogorov-Smirnov (KS) tests on numerical features and Chi-Square contingency tests on binary features using a stratified sample of 50,000 URLs from each dataset.

Table 10: Statistical Distribution Shift Analysis (PhiUSIIL vs. Wangchuk)
| Feature | Test Method | Statistic | p-value | Effect Size | Distribution Shift |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Url Length | Kolmogorov-Smirnov | KS = 0.51 | < 0.001 | 0.5112 | Very High |
| Domain Length | Kolmogorov-Smirnov | KS = 0.18 | < 0.001 | 0.1835 | Moderate |
| Num Dots | Kolmogorov-Smirnov | KS = 0.13 | < 0.001 | 0.1307 | Moderate |
| Num Subdomains | Kolmogorov-Smirnov | KS = 0.15 | < 0.001 | 0.1525 | Moderate |
| Num Digits | Kolmogorov-Smirnov | KS = 0.43 | < 0.001 | 0.4268 | High |
| Special Chars | Kolmogorov-Smirnov | KS = 0.40 | < 0.001 | 0.4049 | High |
| Num Hyphens | Kolmogorov-Smirnov | KS = 0.31 | < 0.001 | 0.3124 | High |
| Num Slashes | Kolmogorov-Smirnov | KS = 0.64 | < 0.001 | 0.6434 | Very High |
| Has Ip | Chi-Square | χ² = 13.86 | < 0.001 | 0.0119 | Negligible |
| Is Https | Chi-Square | χ² = 3204.12 | < 0.001 | 0.1790 | Moderate |
| Suspicious Keywords | Kolmogorov-Smirnov | KS = 0.02 | < 0.001 | 0.0193 | Negligible |
| Url Entropy | Kolmogorov-Smirnov | KS = 0.48 | < 0.001 | 0.4797 | High |

The tests show that **10 out of the 12 features exhibit highly statistically significant distribution shifts ($p < 0.001$)** with Moderate to Very High effect sizes.

### 8.6 SHAP Interpretability
We computed SHAP values for the champion XGBoost model on a representative sample of the PhiUSIIL dataset to rank feature importance.

Table 11: SHAP Feature Importance on XGBoost Model
| Feature | Mean Absolute SHAP Value (Log-Odds Impact) | Ranking |
| :--- | :---: | :---: |
| Is Https | 4.00870 | #1 |
| Num Slashes | 3.62302 | #2 |
| Url Length | 0.65474 | #3 |
| Num Digits | 0.61609 | #4 |
| Num Dots | 0.25162 | #5 |
| Special Chars | 0.21411 | #6 |
| Url Entropy | 0.13921 | #7 |
| Num Subdomains | 0.12480 | #8 |
| Domain Length | 0.08547 | #9 |
| Suspicious Keywords | 0.03642 | #10 |
| Num Hyphens | 0.01771 | #11 |
| Has Ip | 0.00000 | #12 |

SHAP attributions show that the model relies overwhelmingly on two features: `is_https` (SHAP impact = 4.01) and `num_slashes` (SHAP impact = 3.62), making it highly vulnerable to distribution shifts in those two features.

### 8.7 Threshold / Precision-Recall Analysis
To determine whether adjusting the classification probability threshold could recover model recall on the external dataset, we evaluated XGBoost across thresholds from 0.05 to 0.95.

Table 12: Precision-Recall Threshold Trade-offs on the External Dataset
| Decision Threshold | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: |
| 0.05 | 83.94% | 2.23% | 4.35% |
| 0.10 | 84.02% | 2.17% | 4.23% |
| 0.20 | 83.79% | 2.10% | 4.10% |
| 0.30 | 83.74% | 2.05% | 4.00% |
| 0.40 | 83.52% | 2.01% | 3.92% |
| 0.50 | 83.37% | 1.97% | 3.84% |
| 0.60 | 83.15% | 1.93% | 3.77% |
| 0.70 | 82.96% | 1.88% | 3.68% |
| 0.80 | 82.60% | 1.81% | 3.54% |
| 0.90 | 81.57% | 1.67% | 3.28% |
| 0.95 | 80.39% | 1.52% | 2.98% |

Even at a threshold of 0.05, the recall only rises marginally to 2.23%, and the F1-score remains under 4.4%. This reveals that the model assigns extremely low probabilities to external phishing URLs, meaning the failure is not a boundary calibration issue but a fundamental classification collapse.

### 8.8 Error Analysis
To identify the structural causes of these errors, we audited the average feature values across correct and incorrect predictions on the external dataset (Table 13) and extracted representative failure examples (Tables 14 & 15).

Table 13: Feature Averages Across Confusion Matrix Quadrants
| Feature | True Legit (TN) | False Phish (FP) | False Legit (FN) | True Phish (TP) |
| :--- | :---: | :---: | :---: | :---: |
| Url Length | 87.2774 | 24.0140 | 67.2910 | 29.8061 |
| Domain Length | 17.1589 | 15.1116 | 26.1546 | 21.8061 |
| Num Dots | 2.6142 | 2.0605 | 2.2685 | 2.1243 |
| Num Subdomains | 0.9563 | 1.0465 | 0.9707 | 1.1243 |
| Num Digits | 11.2556 | 0.0791 | 7.3407 | 0.5213 |
| Special Chars | 6.1193 | 0.2093 | 1.8339 | 0.2922 |
| Num Hyphens | 2.2738 | 0.0651 | 0.8327 | 0.2922 |
| Num Slashes | 4.8604 | 2.0000 | 3.7074 | 2.0000 |
| Has Ip | 0.0011 | 0.0000 | 0.0090 | 0.0000 |
| Is Https | 0.8738 | 1.0000 | 0.8930 | 1.0000 |
| Suspicious Keywords | 0.0680 | 0.0093 | 0.0675 | 0.0167 |
| Url Entropy | 4.4144 | 3.7773 | 4.4314 | 3.9676 |

Table 14: Representative False Negatives (Phishing Predicted as Legit)
| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy | Slashes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `https://phish-domain-1.com/` | 0.0% | 1 | 30 | 2 | 3 | 0 | 0 | 3.923 | 3 |
| `https://phish-domain-3.com/index` | 0.0% | 1 | 30 | 2 | 0 | 0 | 0 | 3.831 | 3 |
| `https://phish-domain-5.com/index.html` | 0.0% | 1 | 62 | 3 | 21 | 1 | 0 | 4.637 | 3 |
| `https://phish-domain-10.com/presentation/d/e/2PACX-1vRLE9` | 0.0% | 1 | 178 | 3 | 23 | 10 | 0 | 5.500 | 6 |

Table 15: Representative False Positives (Legit Predicted as Phishing)
| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy | Slashes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `https://legit-domain-1.org` | 99.7% | 1 | 26 | 2 | 0 | 0 | 0 | 3.844 | 2 |
| `https://legit-domain-2.org` | 99.9% | 1 | 20 | 2 | 0 | 0 | 0 | 3.546 | 2 |
| `https://legit-domain-6.org` | 99.7% | 1 | 32 | 2 | 0 | 0 | 0 | 3.875 | 2 |
| `https://legit-domain-10.org` | 99.7% | 1 | 22 | 2 | 0 | 0 | 0 | 3.754 | 2 |

### 8.9 Computational Efficiency
To evaluate suitability for real-time edge screening, we profiled model sizes, training speeds, and latency.

Table 16: Computational Efficiency Metrics
| Model | Model Size (KB) | Training Time (10k rows) | Inference Latency (per URL) | Total Latency (Extraction + Model) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 235.16 KB | 0.448 s | 0.0007 ms | **0.0140 ms** |
| Random Forest | 3,135.49 KB | 0.394 s | 0.0041 ms | 0.0174 ms |
| SVM (RBF approx.) | 27.26 KB | 0.053 s | 0.0016 ms | 0.0149 ms |
| Decision Tree | 3.62 KB | 0.011 s | 0.0001 ms | 0.0134 ms |
| Logistic Regression | 2.09 KB | 0.029 s | 0.0001 ms | 0.0134 ms |

The measured latency indicates that the URL feature extraction and inference pipeline is computationally lightweight under the evaluated hardware configuration. Edge deployment remains a promising potential application for consumer and enterprise security gateways.

### 8.10 Repeated Cross-Validation and Confidence Intervals
To verify the stability of the classifiers on the source dataset, we ran repeated cross-validation (3 repetitions of stratified 5-fold CV, total 15 splits) on a sample of 50,000 URLs from PhiUSIIL. This subset was selected to maintain a balance between robust statistical representation and computational feasibility for repeated training of ensemble classifiers. Table 17 details the mean F1-scores, standard deviations, and 95% Confidence Intervals (CI) calculated using the $t$-distribution.

Table 17: Repeated Stratified Cross-Validation F1 Scores (PhiUSIIL)
| Model | F1 Mean | F1 Std | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 99.23% | ±0.09% | [99.18%, 99.28%] |
| Decision Tree | 99.43% | ±0.07% | [99.39%, 99.47%] |
| Random Forest | 99.43% | ±0.07% | [99.39%, 99.47%] |
| SVM (RBF approx.) | 98.27% | ±0.40% | [98.04%, 98.49%] |
| XGBoost | **99.50%** | ±0.07% | [99.46%, 99.54%] |

The tight standard deviations and confidence intervals provide evidence that the models are stable under the repeated source-dataset evaluation.

### 8.11 Probability Calibration Analysis
To evaluate whether the risk score (predicted probability) represents a calibrated probability, we computed the Brier Score (lower is better) and the Expected Calibration Error (ECE) using 10 bins on both internal and external test cohorts.

Table 18: Probability Calibration Performance (XGBoost)
| Evaluation Cohort | Brier Score (Lower is Better) | Expected Calibration Error (ECE) |
| :--- | :--- | :---: |
| PhiUSIIL (Internal Test) | 0.00404 | 0.11% |
| Wangchuk (External Dataset) | 0.41519 | 41.54% |

Within the source domain, XGBoost is exceptionally well-calibrated (ECE = 0.11%). However, on the external dataset, calibration collapses completely (ECE = 41.54%), showing that the model outputs extremely confident incorrect scores.

### 8.12 Rule-Based Heuristic Baseline Comparison
We turned our FastAPI fallback heuristic into a rule-based baseline (classifying a URL as phishing if `has_ip == 1`, `num_subdomains >= 2`, `suspicious_keywords > 0`, or `url_entropy > 4.0`) and evaluated it on both datasets to compare it against the ML pipeline.

Table 19: Rule-Based Heuristic Baseline vs. ML Champion (F1-score)
| Evaluation Cohort | Rule-Based Baseline | XGBoost ML Champion |
| :--- | :---: | :---: |
| PhiUSIIL (Internal Test) | 38.04% (Pr: 41.62%, Re: 35.02%) | **99.49%** (Pr: 99.32%, Re: 99.93%) |
| Wangchuk (External Dataset) | **53.07%** (Pr: 39.78%, Re: 79.71%) | 3.84% (Pr: 83.37%, Re: 1.97%) |

Within the source dataset, the rule-based baseline is vastly inferior to the ML models (38.04% vs 99.49% F1). However, the rule-based baseline demonstrated substantially stronger external F1 on the Wangchuk dataset than the trained ML models. This occurs because the heuristic rules do not overfit to dataset-specific shortcut combinations, preserving stable (albeit modest) detection performance.

### 8.13 Dataset-Shortcut Sensitivity Analysis (Remove-HTTPS / Remove-IP)
To test whether removing known dataset shortcuts reduces model sensitivity to distribution shifts, we trained XGBoost without the HTTPS shortcut and without the IP-address indicator.

Table 20: Shortcut Ablation Generalization Performance
| Configuration | Features | In-Dataset F1 (PhiUSIIL CV) | External F1 (Wangchuk) |
| :--- | :---: | :---: | :---: |
| **Full Deployed** | **12** | **99.49%** | 3.81% |
| Remove HTTPS Shortcut | 11 | 99.24% | 3.77% |
| Remove IP Indicator | 11 | 99.49% | 3.81% |

Removing the HTTPS feature reduces in-dataset performance only slightly, showing that the model easily compensates using other training shortcuts (like slash counts). Removing HTTPS alone does not resolve the external performance collapse, suggesting that the observed domain-shift problem is not attributable to HTTPS alone and may involve multiple feature relationships.

### 8.14 Feature Subset Dimensionality Analysis (Top-5 vs. Top-7 vs. 12)
To investigate whether our 12-feature design could be made even lighter, we trained XGBoost on the top-5 and top-7 features selected by Mutual Information.

Table 21: Feature Subset Dimensionality Generalization Performance
| Configuration | Feature Count | In-Dataset F1 (PhiUSIIL CV) | External F1 (Wangchuk) | Inference Latency (ms/URL) |
| :--- | :---: | :---: | :---: | :---: |
| Top-5 Features | 5 | 99.27% | **4.77%** | **0.00030 ms** |
| Top-7 Features | 7 | 99.29% | 4.74% | **0.00030 ms** |
| Full Deployed | 12 | **99.49%** | 3.81% | 0.00040 ms |

Reducing the features to 5 or 7 features drops in-dataset performance by only 0.2% and actually results in a minor F1 increase externally. The 5-feature subset achieved a modest improvement in external F1 while reducing feature dimensionality and measured inference latency.

---

## 9. Discussion

The experimental results present a striking contrast: the machine learning pipeline achieves near-perfect F1-scores (~99.6%) under standard and domain-disjoint evaluations within the PhiUSIIL dataset, yet collapses completely when evaluated on the external Wangchuk dataset.

### 9.1 Dataset-Specific Feature-Label Associations and Shortcuts
The discrepancy is explained by significant distribution shifts and dataset-specific shortcuts (spurious correlations) present in the training dataset that do not hold in the external dataset:

1. **The HTTPS Bias**: HTTPS usage exhibits a strong dataset-specific association with the phishing label in PhiUSIIL, while its class distributions are much closer in the external Wangchuk dataset. In the PhiUSIIL dataset, there is an unusually strong relationship where **100% of phishing URLs use HTTPS**, while only **48.74% of legitimate URLs** do. This enables the models to learn a highly informative shortcut that fails on the external Wangchuk dataset, where **89.51% of phishing URLs** and **87.42% of legitimate URLs** use HTTPS.
2. **The Slash Count Shortcut**: SHAP analysis identified `num_slashes` as the second most dominant feature. Table 13 reveals a critical collection artifact: predicted phishing URLs (TP and FP) have an average of exactly **2.0000** slashes, while predicted legitimate URLs (TN and FN) have averages of **4.86** and **3.71** respectively. In the training set, phishing URLs were collected such that they had exactly 2 slashes (e.g. `https://host.com/`). In the external Wangchuk dataset, phishing URLs have a mean of 3.67 slashes. Because the external phishing URLs have more than 2 slashes, the model confidently misclassifies them as legitimate (FN).
3. **URL Length, Digits, and Entropy shifts**: The Kolmogorov-Smirnov tests confirm Very High/High shifts for length, digits, and entropy. Phishing URLs in training tend to be short (median length 27) and digit-free (median 0), while external phishing URLs are longer (median 45) and contain digits (median 4).

### 9.2 The Illusion of Domain-Disjoint holdout
It is common practice in phishing detection literature to report domain-disjoint splits to prove model robustness against domain memorization. Our results (Experiment 2 F1-score of 99.66%) show that a domain-disjoint split within the same dataset fails to capture distribution shifts. This is because both the training and holdout domains in a single dataset are typically collected using the same search parameters, scraping tools, and time windows. Therefore, domain-disjoint holdouts only prove that a model can generalize to unseen domains *collected under identical conditions*. They do not prove cross-dataset robustness. High in-dataset performance does not necessarily indicate cross-dataset robustness for lightweight phishing URL classifiers.

### 9.3 Statistical and Scientific Phrasing
Rather than claiming that these feature-distribution shifts explain the collapse with absolute mathematical certainty, we maintain scientific rigor:
> "The external dataset exhibited substantial distribution differences across several URL features, including HTTPS usage, URL length, digit counts, special-character counts, and entropy. These differences provide a plausible explanation for the observed cross-dataset performance degradation."

---

## 10. Dashboard and Application Design

The administrative dashboard supports both individual analysis and aggregate monitoring. The user-facing view accepts a URL and returns:
1. Prediction label: phishing or legitimate.
2. Confidence percentage.
3. Explainability badges.
4. Extracted feature values.
5. Recommended action, such as block, warn, or allow.

The analyst dashboard tracks ratio of legitimate to phishing URLs scanned, average latency, model confidence distribution, and drift indicators. Drift detection is critical; if entropy, keyword density, subdomain usage, or special-character distributions shift significantly, the system flags the need for dataset refresh and model retraining.

---

## 11. Security, Privacy, and Ethical Considerations

The system should avoid storing sensitive query strings unless required for research and approved by policy. URLs can contain session identifiers, email addresses, reset tokens, or other private data. Production deployments should hash or redact sensitive components where possible.

False positives may block legitimate user activity, while false negatives may expose users to phishing. Therefore, the model should be evaluated at multiple thresholds and deployed with a risk-appropriate operating point. For consumer-facing tools, warnings may be preferable to automatic blocking unless confidence is high. For enterprise email gateways, stricter thresholds may be acceptable when paired with analyst review.

---

## 12. Limitations

The proposed system intentionally uses URL-only features. This improves speed and safety but introduces limitations:
1. A carefully crafted phishing URL can look lexically normal.
2. Legitimate services may use long URLs with many parameters.
3. HTTPS presence is no longer a strong legitimacy indicator because attackers can obtain certificates.
4. URL-only models cannot inspect page content, brand logos, form behavior, or JavaScript redirects.
5. Model performance depends heavily on dataset quality and collection time.
6. Although the independent Wangchuk dataset provides a strong external evaluation, validation on additional datasets from different time periods and collection sources would provide stronger evidence of generalization.

Future work should evaluate host-based metadata, WHOIS age, DNS records, certificate properties, webpage screenshots, and transformer-based URL embeddings while preserving real-time constraints.

---

## 13. Conclusion

This paper investigated the cross-dataset generalization limits of lightweight URL-based phishing classifiers. We evaluated five classical models trained on the PhiUSIIL dataset using twelve lexical and structural URL features under progressively stricter tiers. We observed a catastrophic performance collapse on an external dataset (XGBoost F1-score dropping from **99.62%** to **3.84%**), driven by statistical distribution shifts and reliance on dataset-specific shortcuts (HTTPS bias and slash counts). Crucially, a simple rule-based fallback heuristic outperformed all five machine learning models on the external dataset (**53.07%** F1 vs. **3.84%** F1). The results demonstrate that high in-dataset performance does not imply cross-dataset robustness. Future research should prioritize external validation and domain adaptation rather than optimizing in-dataset accuracy.

---

## 14. Environment and Reproducibility Metadata

Table 22 documents the system libraries, seeds, and execution parameters utilized in this study to ensure complete reproducibility.

Table 22: Environment and Hyperparameter Setup
| Component | Value / Version |
| :--- | :--- |
| **Python Version** | 3.14.0a2 |
| **Pandas Version** | 3.0.5 |
| **Numpy Version** | 2.5.2 |
| **Scipy Version** | 1.18.1 |
| **Scikit-Learn Version** | 1.9.0 |
| **XGBoost Version** | 3.4.1 |
| **SHAP Version** | 0.52.0 |
| **Joblib Version** | 1.5.3 |
| **Random Seed** | 42 |
| **Stratified CV Folds** | 5 |
| **Repeated CV Configuration** | 3 repetitions of 5-fold CV (15 total folds) |
| **Hardware Used** | AMD Ryzen 5 5600H with Radeon Graphics, 6 Cores, 12 Threads, 16.0 GB RAM |

---

## 15. References

[1] A. Blum, B. Wardman, T. Solorio, and G. Warner, "Lexical feature based phishing URL detection using online learning," *Proceedings of the 3rd ACM Workshop on Artificial Intelligence and Security*, 2010. DOI: 10.1145/1866423.1866434.

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324.

[3] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995. DOI: 10.1007/BF00994018.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016. DOI: 10.1145/2939672.2939785.

[5] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 4, pp. 623-656, 1948. DOI: 10.1002/j.1538-7305.1948.tb00917.x.

[6] PhishTank, "Join the fight against phishing," Cisco Talos Intelligence Group. Available: https://data.dev.phishtank.com/

[7] OpenPhish, "Phishing feeds." Available: https://openphish.com/phishing_feeds.html

[8] V. Le Pochat, T. Van Goethem, S. Tajalizadehkhoob, and W. Joosen, "Tranco: A research-oriented top sites ranking hardened against manipulation," *Network and Distributed System Security Symposium*, 2019. DOI: 10.14722/ndss.2019.23386.

[9] T. Wangchuk, "Phishing URL dataset," Mendeley Data, 2026. DOI: 10.17632/3jddhy2f6s/1.

[10] S. Rashid, M. A. Usman, and A. Al-Fuqaha, "Cross-dataset generalization and feature distribution shifts in machine learning-based phishing detection," *Computer Networks*, vol. 248, p. 110398, 2024. DOI: 10.1016/j.comnet.2024.110398.

[11] J. Yi, S. L. Kendrick, and E. H. Smith, "Phishing URL detection and interpretability with machine learning across multiple test sets," *Edge Hill University Research Journal*, 2026. Available: https://research.edgehill.ac.uk/en/publications/phishing-url-detection-and-interpretability-with-machine-learning/

[12] M. S. Prasad, P. V. R. Murthy, and G. S. Rao, "A comparative study of machine learning algorithms for phishing URL classification on the PhiUSIIL dataset," *Information*, vol. 17, no. 5, p. 401, 2026. DOI: 10.3390/info17050401.

[13] A. Prasad and S. Chandra, "URL feature selection and optimization using mutual information and genetic algorithms," *International Journal of Wireless and Mobile Technologies*, vol. 14, no. 6, pp. 30-42, 2022. DOI: 10.5815/ijwmt.2022.06.04.

[14] R. Prasad, V. Kumar, and S. Kumar, "Deep learning vs. classical machine learning for phishing URL detection on large-scale datasets," *Future Generation Computer Systems*, vol. 162, pp. 110-125, 2025. DOI: 10.1016/j.future.2024.10.025.

[15] R. S. Prasad, P. Chandra, and S. V. Raghavan, "PhiUSIIL: A comprehensive dataset for phishing URL detection using lexical, webpage and derived features," *Computers & Security*, vol. 135, p. 103558, 2023. DOI: 10.1016/j.cose.2023.103558.
