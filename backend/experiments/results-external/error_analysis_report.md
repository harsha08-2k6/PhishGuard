# Error Analysis Report: External Dataset Failures

## Table 1: Feature Averages Across Confusion Matrix Quadrants

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

## Table 2: Representative False Negatives (Phishing Predicted as Legit)

URLs are obfuscated for security.

| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `https://phish-domain-1.com/` | 0.0% | 1 | 30 | 2 | 3 | 0 | 0 | 3.923 |
| `https://phish-domain-2.com/` | 0.0% | 1 | 47 | 2 | 3 | 0 | 0 | 4.512 |
| `https://phish-domain-3.com/index` | 0.0% | 1 | 30 | 2 | 0 | 0 | 0 | 3.831 |
| `https://phish-domain-4.com/ipfs/bafybeifkfupfhxt6rr72yjn` | 0.0% | 1 | 109 | 4 | 10 | 0 | 0 | 4.799 |
| `https://phish-domain-5.com/index.html` | 0.0% | 1 | 62 | 3 | 21 | 1 | 0 | 4.637 |
| `http://phish-domain-6.com/v3/signin/identifier?dsh=S144` | 0.0% | 0 | 460 | 13 | 88 | 28 | 2 | 5.720 |
| `https://phish-domain-7.com/` | 0.0% | 1 | 31 | 2 | 6 | 1 | 0 | 4.015 |
| `https://phish-domain-8.com/ipfs/bafybeibnczkxh6gtu5cbvpv` | 0.0% | 1 | 92 | 1 | 6 | 1 | 0 | 4.823 |
| `https://phish-domain-9.com/emergencyrelief/apply` | 0.0% | 1 | 46 | 2 | 0 | 0 | 0 | 4.059 |
| `https://phish-domain-10.com/presentation/d/e/2PACX-1vRLE9` | 0.0% | 1 | 178 | 3 | 23 | 10 | 0 | 5.500 |

## Table 3: Representative False Positives (Legit Predicted as Phishing)

URLs are obfuscated for security.

| Obfuscated URL | Prob. | HTTPS | Length | Dots | Digits | Special | Keywords | Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `https://legit-domain-1.org` | 99.7% | 1 | 26 | 2 | 0 | 0 | 0 | 3.844 |
| `https://legit-domain-2.org` | 99.9% | 1 | 20 | 2 | 0 | 0 | 0 | 3.546 |
| `https://legit-domain-3.org` | 99.9% | 1 | 22 | 3 | 0 | 0 | 0 | 3.754 |
| `https://legit-domain-4.org?lang=en...` | 92.4% | 1 | 36 | 2 | 0 | 2 | 0 | 4.083 |
| `https://legit-domain-5.org` | 99.9% | 1 | 19 | 2 | 0 | 0 | 1 | 3.366 |
| `https://legit-domain-6.org` | 99.7% | 1 | 32 | 2 | 0 | 0 | 0 | 3.875 |
| `https://legit-domain-7.org` | 99.7% | 1 | 22 | 2 | 0 | 0 | 0 | 3.754 |
| `https://legit-domain-8.org` | 99.6% | 1 | 24 | 2 | 0 | 0 | 0 | 3.939 |
| `https://legit-domain-9.org` | 99.8% | 1 | 21 | 2 | 0 | 0 | 0 | 3.559 |
| `https://legit-domain-10.org` | 99.7% | 1 | 22 | 2 | 0 | 0 | 0 | 3.754 |