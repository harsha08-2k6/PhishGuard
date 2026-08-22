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

## 8. Benchmark Reporting Template

The following table should be populated after training and testing on the selected dataset. Placeholder values are intentionally omitted to avoid reporting unverified performance.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |

## 9. Dashboard and Application Design

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

## 10. Security, Privacy, and Ethical Considerations

The system should avoid storing sensitive query strings unless required for research and approved by policy. URLs can contain session identifiers, email addresses, reset tokens, or other private data. Production deployments should hash or redact sensitive components where possible.

False positives may block legitimate user activity, while false negatives may expose users to phishing. Therefore, the model should be evaluated at multiple thresholds and deployed with a risk-appropriate operating point. For consumer-facing tools, warnings may be preferable to automatic blocking unless confidence is high. For enterprise email gateways, stricter thresholds may be acceptable when paired with analyst review.

The system should not be represented as a complete replacement for browser safe-browsing services, DNS filtering, or user education. It is best understood as a fast URL-level detection layer within a broader defense-in-depth strategy.

## 11. Limitations

The proposed system intentionally uses URL-only features. This improves speed and safety but introduces limitations:

1. A carefully crafted phishing URL can look lexically normal.
2. Legitimate services may use long URLs with many parameters.
3. HTTPS presence is no longer a strong legitimacy indicator because attackers can obtain certificates.
4. URL-only models cannot inspect page content, brand logos, form behavior, or JavaScript redirects.
5. Model performance depends heavily on dataset quality and collection time.

Future work should evaluate host-based metadata, WHOIS age, DNS records, certificate properties, webpage screenshots, and transformer-based URL embeddings while preserving real-time constraints.

## 12. Conclusion

This paper presented a complete design for an automated phishing URL detection and explainability system based on lexical machine learning features. The system extracts twelve lightweight indicators from a submitted URL, compares multiple supervised classifiers, outputs a phishing probability, and provides human-readable diagnostic flags. Its URL-only design supports real-time operation without visiting potentially malicious websites. The proposed benchmark protocol, dashboard design, and ethical safeguards make the system suitable as both an academic research project and a practical cybersecurity application prototype.

## References

[1] A. Blum, B. Wardman, T. Solorio, and G. Warner, "Lexical feature based phishing URL detection using online learning," *Proceedings of the 3rd ACM Workshop on Artificial Intelligence and Security*, 2010. DOI: 10.1145/1866423.1866434.

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324.

[3] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995. DOI: 10.1007/BF00994018.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016. DOI: 10.1145/2939672.2939785.

[5] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 4, pp. 623-656, 1948. DOI: 10.1002/j.1538-7305.1948.tb00917.x.

[6] PhishTank, "Join the fight against phishing," Cisco Talos Intelligence Group. Available: https://data.dev.phishtank.com/

[7] OpenPhish, "Phishing feeds." Available: https://openphish.com/phishing_feeds.html

[8] V. Le Pochat, T. Van Goethem, S. Tajalizadehkhoob, and W. Joosen, "Tranco: A research-oriented top sites ranking hardened against manipulation," *Network and Distributed System Security Symposium*, 2019. DOI: 10.14722/ndss.2019.23386.
