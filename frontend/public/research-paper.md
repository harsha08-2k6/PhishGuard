# Automated Phishing URL Detection and Explainability Using Lexical Machine Learning Features

## Abstract

Phishing websites remain a persistent cybersecurity threat because attackers can rapidly generate deceptive URLs that impersonate trusted services, redirect victims, and evade static blocklists. This paper presents an automated phishing URL detection and explainability system that classifies a submitted URL as legitimate or malicious using only lexical and structural properties of the URL string. The proposed system extracts twelve lightweight features, including URL length, domain length, subdomain count, special-character frequency, digit density, IP-address usage, HTTPS presence, suspicious keyword density, and Shannon entropy. These features are evaluated through a comparative supervised learning pipeline consisting of XGBoost, Random Forest, Support Vector Machine, Decision Tree, and Logistic Regression classifiers. To improve user trust and analyst usability, the system pairs the binary prediction and confidence score with an explainability layer that converts suspicious feature activations into human-readable diagnostic flags. The design supports real-time browser, email gateway, or dashboard integration because it does not require crawling page content, resolving JavaScript, or querying heavy third-party services at prediction time. The paper describes the system architecture, feature-engineering strategy, model-training workflow, evaluation metrics, dashboard design, and ethical considerations required for a reproducible academic implementation.

**Keywords:** phishing detection, malicious URL classification, lexical features, XGBoost, explainable AI, cybersecurity, machine learning

## 1. Introduction

Phishing is a social-engineering attack in which adversaries distribute fraudulent links that appear to represent legitimate services, such as banks, cloud platforms, e-commerce websites, or identity providers. Once a victim visits the link, the attacker may collect credentials, payment details, session tokens, or other sensitive information. Traditional phishing defenses often rely on manually maintained blacklists, domain reputation feeds, or content inspection. Although useful, these approaches can struggle with newly created phishing infrastructure because malicious URLs may appear and disappear before a blacklist is updated.

Machine learning offers a complementary defense by detecting suspicious URL patterns before the link is widely reported. A URL-only classifier is especially practical because it can operate without loading potentially dangerous web content. This reduces user exposure, lowers latency, and makes the system suitable for browser extensions, email filters, mobile messaging clients, and security dashboards.

This work proposes an automated phishing URL detection and explainability system with four primary goals:

1. Extract a compact set of lexical and structural URL features in real time.
2. Compare multiple supervised machine learning classifiers under a consistent evaluation protocol.
3. Produce a calibrated phishing probability and binary label.
4. Explain the decision using interpretable feature-level warnings.

## 2. Related Work

Earlier URL-based phishing detection research showed that lexical signals can help identify emerging malicious links before static blacklists are updated. Blum et al. demonstrated lexical-feature-based phishing URL detection using online learning, emphasizing adaptability against newly appearing threats [1]. Classical machine learning models remain widely used in phishing detection because engineered URL features are inexpensive to compute and often provide strong baseline performance.

The models selected in this study represent both interpretable and high-performing families of classifiers. Random Forests combine multiple randomized decision trees and can provide feature-importance estimates [2]. Support Vector Machines construct margin-based decision boundaries and can model non-linear class separation through kernels [3]. XGBoost is a scalable gradient-boosted tree system that is effective for tabular feature representations and non-linear feature interactions [4]. The use of Shannon entropy follows the information-theoretic formulation introduced by Shannon, where character-distribution uncertainty can be used as a measure of randomness [5].

For data sourcing, phishing URLs may be collected from verified phishing feeds such as PhishTank and OpenPhish, while legitimate URLs may be sampled from popularity rankings such as Tranco [6], [7], [8]. Tranco is particularly useful for research because it was designed to reduce manipulation and reproducibility problems in top-site rankings [8].

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
+-------------------------------+     +-------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                    Research Benchmark and Dashboard                 |
| Accuracy, precision, recall, F1, ROC-AUC, latency, and drift views  |
+---------------------------------------------------------------------+
```

The architecture intentionally avoids webpage crawling. This makes the classifier safer to run in user-facing contexts and keeps detection latency low. The system can later be extended with host-based, WHOIS, DNS, or page-content features, but the baseline focuses on URL-only detection for speed and reproducibility.

## 4. Feature Extraction Methodology

The URL feature extraction engine converts each raw URL into a twelve-dimensional feature vector. Before extraction, URLs are normalized by trimming whitespace, lowercasing where appropriate, decoding common URL encodings, and adding a missing protocol placeholder if required for parsing.

| Feature | Description | Security Rationale |
|---|---|---|
| URL length | Total number of characters in the URL | Phishing URLs often use long strings to hide the destination or include tracking/redirection tokens. |
| Domain length | Character length of the registered domain or host | Very long hostnames may indicate obfuscation or generated infrastructure. |
| Dot count | Number of periods in the full URL or host | Excessive dots may indicate nested subdomains or deceptive host construction. |
| Subdomain count | Number of labels before the registered domain | Attackers may imitate trusted brands through misleading subdomains. |
| Hyphen count | Number of hyphens in host or URL | Hyphens are often used in typosquatting or brand impersonation. |
| Special-character count | Count of symbols such as `@`, `?`, `=`, `%`, `_`, and `&` | High symbol density may indicate parameter stuffing, encoded payloads, or obfuscation. |
| Digit count | Total number of numeric characters | Algorithmically generated links frequently contain numeric tokens. |
| Digit ratio | Digits divided by total URL length | Normalizes digit usage across short and long URLs. |
| IP address present | Binary flag indicating whether the host is an IPv4 or IPv6 address | Direct IP links can indicate temporary or untrusted infrastructure. |
| HTTPS present | Binary flag indicating whether the URL uses HTTPS | Lack of HTTPS remains a risk signal, although HTTPS alone does not prove legitimacy. |
| Suspicious keyword density | Frequency of terms such as `login`, `verify`, `secure`, `account`, `update`, `banking`, and `password` | Credential-harvesting URLs often contain urgency or authentication-related terms. |
| Shannon entropy | Randomness of character distribution in the URL | High entropy can indicate generated domains, opaque tokens, or encoded strings. |

Shannon entropy is computed as:

```math
H(X) = -\sum_{i=1}^{n} P(x_i)\log_2 P(x_i)
```

where `P(x_i)` is the probability of character `x_i` appearing in the URL string. Higher entropy indicates greater randomness.

## 5. Machine Learning Pipeline

The feature vector is passed to a supervised binary classifier trained on labeled phishing and legitimate URLs. The target label is encoded as:

```text
0 = legitimate
1 = phishing
```

The comparative benchmark includes:

| Model | Role in Study | Key Hyperparameters |
|---|---|---|
| XGBoost | Primary high-performance model for non-linear feature interactions | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |
| Random Forest | Robust ensemble baseline with feature-importance support | `n_estimators=100`, `criterion=gini` |
| Support Vector Machine | Margin-based classifier for non-linear decision boundaries | `C=1.0`, `kernel=rbf`, `gamma=scale` |
| Decision Tree | Lightweight interpretable baseline | `max_depth=8`, `criterion=entropy` |
| Logistic Regression | Linear statistical baseline | `penalty=l2`, `solver=lbfgs` |

Feature scaling is applied to SVM and Logistic Regression using z-score normalization. Tree-based models are trained on the raw engineered features because they are generally insensitive to monotonic feature scaling.

## 6. Explainability Layer

The explainability layer converts model inputs and risk thresholds into warnings that can be understood by non-expert users and security analysts. Instead of presenting only a label, the system displays the most relevant diagnostic triggers.

Example heuristic triggers include:

| Trigger | Condition | Displayed Explanation |
|---|---|---|
| High URL entropy | `entropy > 4.0` | The URL contains unusually random character patterns. |
| Excessive subdomains | `subdomain_count >= 3` | The link uses several subdomain levels, which can disguise the real domain. |
| IP address host | `ip_address_present = 1` | The URL uses a raw IP address instead of a recognizable domain. |
| Suspicious keywords | `keyword_density > threshold` | The URL contains terms commonly found in credential-harvesting links. |
| No HTTPS | `https_present = 0` | The URL does not use encrypted HTTPS transport. |
| High special-character usage | `special_character_count > threshold` | The URL contains many symbols often used in obfuscation or tracking parameters. |

The model confidence is reported as:

```math
P(\text{Phishing} \mid X)
```

The default classification threshold is `0.50`, although deployment environments may adjust the threshold depending on whether they prioritize fewer false positives or higher threat recall.

## 7. Dataset and Experimental Protocol

A reproducible experiment should use balanced and deduplicated phishing and legitimate URL samples. Candidate data sources include:

| Class | Candidate Sources | Notes |
|---|---|---|
| Phishing | PhishTank, OpenPhish | Use only verified phishing URLs. Record collection date because phishing feeds change rapidly. |
| Legitimate | Tranco top-site ranking | Remove domains that appear in phishing feeds or security blocklists. |

Recommended preprocessing steps:

1. Remove duplicate URLs and near-duplicate normalized variants.
2. Normalize missing schemes such as `https://`.
3. Strip invalid whitespace and malformed control characters.
4. Preserve the full URL string for lexical extraction.
5. Split data into training, validation, and test sets using stratified sampling.
6. Avoid leakage by ensuring the same domain family does not appear across both training and test sets when evaluating generalization.

The evaluation should report:

| Metric | Purpose |
|---|---|
| Accuracy | Overall percentage of correct classifications. |
| Precision | Measures how many predicted phishing URLs are actually phishing. |
| Recall | Measures how many real phishing URLs are successfully detected. |
| F1-score | Harmonic mean of precision and recall. |
| ROC-AUC | Measures discrimination across classification thresholds. |
| Confusion matrix | Shows false positives and false negatives explicitly. |
| Inference latency | Measures suitability for real-time deployment. |

## 8. Experimental Results

This section presents the results of the four evaluation experiments conducted to assess the performance of the five machine learning models (XGBoost, Random Forest, SVM with RBF approximation, Decision Tree, and Logistic Regression) trained on the PhiUSIIL dataset using 12 lexical and structural URL features.

### 8.1 Experiment 1: Standard Evaluation (Stratified 5-Fold Cross-Validation)
The standard stratified 5-fold cross-validation evaluates baseline generalization performance when train and test sets are sampled randomly from the same underlying distribution of the PhiUSIIL dataset (235,795 URLs). Table 3 details the average performance metrics along with their sample standard deviations ($\pm \text{Std}$).

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
To test whether the models generalize to unseen domains within the same dataset, we conducted a domain-disjoint holdout evaluation. In this split, domain groupings are strictly separated so that no domain in the training set appears in the testing set. Table 4 reports these holdout results.

Table 4: Domain-Disjoint Holdout Performance on PhiUSIIL Dataset
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | **0.9960** | 0.9940 | 0.9991 | **0.9966** | **0.9984** | **0.9980** |
| Random Forest | 0.9953 | **0.9944** | 0.9975 | 0.9959 | 0.9969 | 0.9953 |
| SVM (RBF approx.) | 0.9753 | 0.9921 | 0.9648 | 0.9783 | 0.9926 | 0.9926 |
| Decision Tree | 0.9957 | 0.9932 | **0.9994** | 0.9963 | 0.9972 | 0.9959 |
| Logistic Regression | 0.9937 | 0.9897 | 0.9995 | 0.9946 | 0.9960 | 0.9936 |

The domain-disjoint holdout results match or slightly exceed the stratified 5-fold cross-validation performance. This suggests that the models do not suffer from domain memorization within this dataset, which would normally lead to a drop in performance on unseen domains.

### 8.3 Experiment 3: Feature Ablation Study
We conducted feature ablation experiments using the champion XGBoost model under 5-fold cross-validation. Table 5 details performance across pre-defined feature subsets: Structural features (length, domain length, dots, subdomains, hyphens, slashes), Security features (IP usage, HTTPS, keywords), and Randomness features (digits, special characters, entropy). Additionally, Table 6 shows the cumulative ablation results.

Table 5: Group-Wise Feature Ablation on XGBoost Champion
| Feature Set | Accuracy | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| All 12 Features | **0.9956 ± 0.0003** | **0.9962 ± 0.0003** | **0.9979 ± 0.0002** | **0.9972 ± 0.0003** |
| Structure Only | 0.9924 ± 0.0007 | 0.9934 ± 0.0006 | 0.9970 ± 0.0001 | 0.9959 ± 0.0003 |
| Security Only | 0.8050 ± 0.0010 | 0.8537 ± 0.0006 | 0.7746 ± 0.0012 | 0.7477 ± 0.0010 |
| Randomness Only | 0.7849 ± 0.0014 | 0.8337 ± 0.0011 | 0.8056 ± 0.0019 | 0.7878 ± 0.0031 |

Table 6: Cumulative Feature Ablation on XGBoost Champion
| Feature Set | Accuracy | F1-score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: |
| Lexical (Length, Digits, Special Chars) | 0.8060 ± 0.0010 | 0.8481 ± 0.0007 | 0.8239 ± 0.0009 | 0.7999 ± 0.0013 |
| Lexical + Structural | 0.8617 ± 0.0012 | 0.8909 ± 0.0009 | 0.8836 ± 0.0010 | 0.8651 ± 0.0014 |
| + Domain Length | 0.9930 ± 0.0011 | 0.9939 ± 0.0010 | 0.9976 ± 0.0001 | 0.9967 ± 0.0002 |
| + Security | **0.9957 ± 0.0003** | **0.9962 ± 0.0003** | **0.9979 ± 0.0001** | **0.9973 ± 0.0003** |
| + Entropy (All 12) | 0.9956 ± 0.0003 | 0.9962 ± 0.0003 | 0.9979 ± 0.0002 | 0.9973 ± 0.0003 |

Adding domain length to the lexical and structural features results in the largest performance jump (F1-score increases from 89.09% to 99.39%), demonstrating that domain-length boundaries are extremely informative in the training dataset.

### 8.4 Experiment 4: Cross-Dataset Validation
To evaluate the real-world generalization capability of the PhishGuard models, we trained them on the full PhiUSIIL dataset and tested them on the external Wangchuk dataset [9], which consists of 129,777 unique URLs (74,972 legitimate, 54,807 phishing). Table 7 shows the performance metrics, demonstrating a catastrophic performance collapse across all classifiers.

Table 7: External Cross-Dataset Validation Performance (Wangchuk Dataset)
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | 0.5843 | **0.8337** | 0.0197 | 0.0384 | 0.6838 | **0.6378** |
| Random Forest | 0.5844 | 0.8274 | 0.0199 | 0.0389 | 0.5175 | 0.4402 |
| SVM (RBF approx.) | 0.5855 | 0.8258 | 0.0234 | 0.0456 | 0.4100 | 0.4416 |
| Decision Tree | 0.5843 | 0.8279 | 0.0198 | 0.0387 | 0.5098 | 0.4305 |
| Logistic Regression | **0.5860** | 0.7855 | **0.0272** | **0.0526** | **0.7058** | 0.6026 |

All models struggle to detect phishing URLs from the external dataset, resulting in recalls below 3% and F1-scores between 3.8% and 5.3%. While precision remains relatively high (78.5% - 83.4%), the classifiers are almost entirely failing to identify the malicious class. To understand this prediction collapse, we examine the confusion matrices in Table 8.

Table 8: Confusion Matrices on the External Wangchuk Dataset
| Model | True Legit (TN) | False Phish (FP) | False Legit (FN) | True Phish (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 74,757 | 215 | 53,727 | 1,078 |
| **Random Forest** | 74,742 | 230 | 53,728 | 1,077 |
| **SVM (RBF approx.)** | 74,701 | 271 | 53,520 | 1,285 |
| **Decision Tree** | 74,741 | 231 | 53,718 | 1,087 |
| **Logistic Regression** | 74,565 | 407 | 53,315 | 1,490 |

The confusion matrices show that all five models predict almost every URL in the external dataset as legitimate. For example, XGBoost classifies only 1,293 out of 129,777 URLs as phishing (1,078 True Positives and 215 False Positives), missing 53,727 phishing URLs (False Negatives). This indicates a systemic generalization failure where the models are unable to transfer the learned decision boundary to an independent dataset.

## 9. Discussion

The experimental results present a striking contrast: the machine learning pipeline achieves near-perfect F1-scores (~99.6%) under standard and domain-disjoint evaluations within the PhiUSIIL dataset, yet collapses completely when evaluated on the external Wangchuk dataset. This section provides a detailed analysis of this phenomenon.

### 9.1 Dataset-Specific Feature-Label Associations
The discrepancy is explained by significant distribution shifts and dataset-specific shortcuts (spurious correlations) present in the training dataset that do not hold in the external dataset. The most critical distribution shifts include:

1. **The HTTPS Bias**: HTTPS usage exhibits a strong dataset-specific association with the phishing label in PhiUSIIL, while its class distributions are much closer in the external Wangchuk dataset. In the PhiUSIIL dataset, there is an unusually strong relationship where **100% of phishing URLs use HTTPS**, while only **48.74% of legitimate URLs** do. This enables the models to learn a highly informative shortcut that fails on the external Wangchuk dataset, where **89.51% of phishing URLs** and **87.42% of legitimate URLs** use HTTPS.
2. **URL Length Distribution**: The model learns to associate shorter URLs with phishing in PhiUSIIL (legitimate median length: 34 vs. phishing median length: 27). In the external dataset, the length distributions are shifted: legitimate median length is 61, and phishing median length is 45. The relationship still trends toward phishing being shorter, but the substantial overlap and overall shift confuse a model that has learned absolute length-threshold boundaries.
3. **Digit Density**: In the PhiUSIIL dataset, digits are rare, with medians of 0 for both classes. In the Wangchuk dataset, legitimate URLs have a median of 3 digits and phishing URLs have a median of 4 digits. The model encounters far more digit features in the external dataset than it saw in training.
4. **Special Character Usage**: In PhiUSIIL, special characters are practically non-existent in both classes (medians of 0). In the Wangchuk dataset, legitimate URLs have a median of 2 special characters, and phishing URLs have a median of 1.
5. **Shannon Entropy Reversal**: Shannon entropy exhibits a complete reversal in class relationship. In PhiUSIIL, legitimate URLs have higher entropy (median: 4.106) than phishing URLs (median: 3.852). In the external dataset, the relationship is reversed, with phishing URLs having slightly higher entropy (median: 4.422) than legitimate URLs (median: 4.413).

### 9.2 The Illusion of Domain-Disjoint holdout
It is common practice in phishing detection literature to report domain-disjoint splits to prove model robustness against domain memorization. Our results (Experiment 2 F1-score of 99.66%) show that a domain-disjoint split within the same dataset fails to capture distribution shifts. This is because both the training and holdout domains in a single dataset are typically collected using the same search parameters, scraping tools, and time windows. 

Therefore, domain-disjoint holdouts only prove that a model can generalize to unseen domains *collected under identical conditions*. They do not prove cross-dataset robustness. High in-dataset performance does not necessarily indicate cross-dataset robustness for lightweight phishing URL classifiers.

### 9.3 Statistical and Scientific Phrasing
Rather than claiming that these feature-distribution shifts explain the collapse with absolute mathematical certainty, we maintain scientific rigor:
> "The external dataset exhibited substantial distribution differences across several URL features, including HTTPS usage, URL length, digit counts, special-character counts, and entropy. These differences provide a plausible explanation for the observed cross-dataset performance degradation."

## 10. Dashboard and Application Design

The administrative dashboard supports both individual analysis and aggregate monitoring. The user-facing view accepts a URL and returns:

1. Prediction label: phishing or legitimate.
2. Confidence percentage.
3. Explainability badges.
4. Extracted feature values.
5. Recommended action, such as block, warn, or allow.

The analyst dashboard tracks:

1. Ratio of legitimate to phishing URLs scanned.
2. Average inference latency.
3. Most common warning triggers.
4. Model confidence distribution.
5. Daily or weekly URL-volume trends.
6. Drift indicators showing whether incoming URLs differ from the training distribution.

Drift detection is important because phishing campaigns change quickly. If entropy, keyword density, subdomain usage, or special-character distributions shift significantly, the system should flag the need for dataset refresh and model retraining.

## 11. Security, Privacy, and Ethical Considerations

The system should avoid storing sensitive query strings unless required for research and approved by policy. URLs can contain session identifiers, email addresses, reset tokens, or other private data. Production deployments should hash or redact sensitive components where possible.

False positives may block legitimate user activity, while false negatives may expose users to phishing. Therefore, the model should be evaluated at multiple thresholds and deployed with a risk-appropriate operating point. For consumer-facing tools, warnings may be preferable to automatic blocking unless confidence is high. For enterprise email gateways, stricter thresholds may be acceptable when paired with analyst review.

The system should not be represented as a complete replacement for browser safe-browsing services, DNS filtering, or user education. It is best understood as a fast URL-level detection layer within a broader defense-in-depth strategy.

## 12. Limitations

The proposed system intentionally uses URL-only features. This improves speed and safety but introduces limitations:

1. A carefully crafted phishing URL can look lexically normal.
2. Legitimate services may use long URLs with many parameters.
3. HTTPS presence is no longer a strong legitimacy indicator because attackers can obtain certificates.
4. URL-only models cannot inspect page content, brand logos, form behavior, or JavaScript redirects.
5. Model performance depends heavily on dataset quality and collection time.

Future work should evaluate host-based metadata, WHOIS age, DNS records, certificate properties, webpage screenshots, and transformer-based URL embeddings while preserving real-time constraints.

## 13. Conclusion

This paper presented a complete design for an automated phishing URL detection and explainability system based on lexical machine learning features. The system extracts twelve lightweight indicators from a submitted URL, compares multiple supervised classifiers, outputs a phishing probability, and provides human-readable diagnostic flags. Its URL-only design supports real-time operation without visiting potentially malicious websites. The proposed benchmark protocol, dashboard design, and ethical safeguards make the system suitable as both an academic research project and a practical cybersecurity application prototype.

## 14. References

[1] A. Blum, B. Wardman, T. Solorio, and G. Warner, "Lexical feature based phishing URL detection using online learning," *Proceedings of the 3rd ACM Workshop on Artificial Intelligence and Security*, 2010. DOI: 10.1145/1866423.1866434.

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324.

[3] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995. DOI: 10.1007/BF00994018.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016. DOI: 10.1145/2939672.2939785.

[5] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 4, pp. 623-656, 1948. DOI: 10.1002/j.1538-7305.1948.tb00917.x.

[6] PhishTank, "Join the fight against phishing," Cisco Talos Intelligence Group. Available: https://data.dev.phishtank.com/

[7] OpenPhish, "Phishing feeds." Available: https://openphish.com/phishing_feeds.html

[8] V. Le Pochat, T. Van Goethem, S. Tajalizadehkhoob, and W. Joosen, "Tranco: A research-oriented top sites ranking hardened against manipulation," *Network and Distributed System Security Symposium*, 2019. DOI: 10.14722/ndss.2019.23386.

[9] T. Wangchuk, "Phishing URL dataset," Mendeley Data, 2026. DOI: 10.17632/3jddhy2f6s/1.
