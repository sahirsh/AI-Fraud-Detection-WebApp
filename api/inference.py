import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocess import clean_text

MODEL_PATH = "ml/model.joblib"
VECTORIZER_PATH = "ml/vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

LOW_RISK = 0.3
MEDIUM_RISK = 0.6
HIGH_RISK = 0.7

def predict_job_fraud(title = "", company_profile = "", description = "", requirements = "", benefits = ""):
    """
    Predicts the fraud risk of a job posting based on its text fields.

    Returns:
    - dict: A dictionary containing the fraud probability and risk level.
    """
    
    text_fused = " ".join([
        title or "", company_profile or "", description or "",
        requirements or "", benefits or ""
    ])

    cleaned_text = clean_text(text_fused)
    X = vectorizer.transform([cleaned_text])

    fraud_prob = model.predict_proba(X)[0][1]

    if fraud_prob >= HIGH_RISK:
        label = 1
        risk = "High RISK(Likely Fraudulent)"
    elif fraud_prob >= MEDIUM_RISK:
        label = 1
        risk = "Medium Risk(Potentially Fraudulent)"
    elif fraud_prob >= LOW_RISK:
        label = 0
        risk = "Low Risk(Unlikely Fraudulent)"
    else:
        label = 0
        risk = "Very Low Risk(Likely Safe)"

    return {
        "fraud_probability" : round(float(fraud_prob), 4),
        "label": label,
        "risk_level": risk
    }


