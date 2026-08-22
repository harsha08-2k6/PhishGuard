import math
import re
from urllib.parse import urlparse


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "token",
    "free",
]
IP_PATTERN = r"(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])"


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [text.count(char) / len(text) for char in dict.fromkeys(text)]
    return -sum(probability * math.log2(probability) for probability in probabilities)


def extract_features(url: str) -> dict:
    normalized_url = url if "://" in url else f"http://{url}"
    parsed = urlparse(normalized_url)
    domain = parsed.netloc
    lower_url = url.lower()

    return {
        "url_length": len(url),
        "domain_length": len(domain),
        "num_dots": url.count("."),
        "num_subdomains": max(0, domain.count(".") - 1),
        "num_digits": sum(char.isdigit() for char in url),
        "special_chars": len(re.findall(r"[-_?=&%~#]", url)),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "has_ip": int(bool(re.search(IP_PATTERN, domain))),
        "is_https": int(parsed.scheme == "https"),
        "suspicious_keywords": sum(word in lower_url for word in SUSPICIOUS_WORDS),
        "url_entropy": round(calculate_entropy(url), 3),
    }


def explain_prediction(features: dict) -> list[str]:
    reasons = []
    if features["has_ip"]:
        reasons.append("Host is an unresolved raw IP address.")
    if features["num_subdomains"] >= 3:
        reasons.append(f"Excessive subdomains detected ({features['num_subdomains']}).")
    if features["url_entropy"] > 4.2:
        reasons.append(f"High character randomness/entropy ({features['url_entropy']}).")
    if features["suspicious_keywords"]:
        reasons.append(
            "Contains high-risk authentication keywords "
            f"({features['suspicious_keywords']} found)."
        )
    if features["special_chars"] > 5:
        reasons.append("Heavy use of parameter obfuscation characters.")
    if not features["is_https"]:
        reasons.append("Unencrypted connection (HTTP).")
    return reasons or ["Lexical structure aligns with normal domain profiles."]
