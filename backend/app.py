import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extract_features import explain_prediction, extract_features


MODEL_PATHS = {
    "XGBoost": Path(__file__).parent / "models" / "xgboost_phish.pkl",
    "Random Forest": Path(__file__).parent / "models" / "random_forest_phish.pkl",
    "SVM (RBF approximation)": Path(__file__).parent / "models" / "svm_rbf_approximation.pkl",
    "Decision Tree": Path(__file__).parent / "models" / "decision_tree_phish.pkl",
    "Logistic Regression": Path(__file__).parent / "models" / "logistic_regression_phish.pkl",
}
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
        "XGBoost", "Random Forest", "SVM (RBF approximation)", "Decision Tree", "Logistic Regression"
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
    model_path = MODEL_PATHS[model_name]
    if model_path.exists():
        try:
            import joblib

            model = joblib.load(model_path)
            values = [[features[key] for key in features]]
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(values)[0][1])
            else:
                decision_value = float(model.decision_function(values)[0])
                probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, decision_value))))
            return probability, "trained-model"
        except (ImportError, OSError, ValueError, AttributeError, IndexError):
            pass

    if model_name != "XGBoost":
        raise FileNotFoundError(
            f"No trained artifact is available for {model_name}. "
            "Run the experiment runner with --save-all-models first."
        )

    base = heuristic_score(features)
    return min(0.99, max(0.01, base)), "demo-heuristic"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "available_models": [name for name, path in MODEL_PATHS.items() if path.exists()],
    }


@app.post("/api/scan")
def scan(request: ScanRequest) -> dict:
    features = extract_features(request.url)
    try:
        probability, inference_mode = model_score(features, request.model)
    except FileNotFoundError as error:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "url": request.url,
        "model": request.model,
        "risk_score": round(probability * 100, 1),
        "verdict": "phishing" if probability >= 0.5 else "legitimate",
        "features": features,
        "reasons": explain_prediction(features),
        "inference_mode": inference_mode,
    }
