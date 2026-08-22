from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extract_features import explain_prediction, extract_features


MODEL_PATH = Path(__file__).parent / "models" / "xgboost_phish.pkl"
app = FastAPI(title="PhishGuard Research API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    url: str = Field(min_length=3, max_length=4096)
    model: Literal[
        "XGBoost", "Random Forest", "SVM", "Decision Tree", "Logistic Regression"
    ] = "XGBoost"


def heuristic_score(features: dict) -> float:
    score = 0.15
    if features["has_ip"]:
        score += 0.35
    if features["num_subdomains"] >= 2:
        score += 0.20
    if features["suspicious_keywords"]:
        score += 0.20
    if features["url_entropy"] > 4.0:
        score += 0.15
    return min(0.99, max(0.01, score))


def model_score(features: dict, model_name: str) -> tuple[float, str]:
    if MODEL_PATH.exists():
        try:
            import joblib

            model = joblib.load(MODEL_PATH)
            values = [[features[key] for key in features]]
            probability = float(model.predict_proba(values)[0][1])
            return probability, "trained-model"
        except (ImportError, OSError, ValueError, AttributeError, IndexError):
            pass

    profile = {
        "XGBoost": (1.0, 0.0),
        "Random Forest": (0.96, 0.02),
        "SVM": (0.92, 0.04),
        "Decision Tree": (1.05, -0.01),
        "Logistic Regression": (0.88, 0.05),
    }[model_name]
    base = heuristic_score(features)
    return min(0.99, max(0.01, base * profile[0] + profile[1])), "heuristic-proxy"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/api/scan")
def scan(request: ScanRequest) -> dict:
    features = extract_features(request.url)
    probability, inference_mode = model_score(features, request.model)
    return {
        "url": request.url,
        "model": request.model,
        "risk_score": round(probability * 100, 1),
        "verdict": "phishing" if probability >= 0.5 else "legitimate",
        "features": features,
        "reasons": explain_prediction(features),
        "inference_mode": inference_mode,
    }
