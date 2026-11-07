
# ==========================================================
# DAY 23 - CREDIT CARD FRAUD DETECTION API
# FastAPI + Pydantic + Saved ML Model
# ==========================================================

import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import create_model, ConfigDict


# ----------------------------------------------------------
# 1. LOCATE AND LOAD THE SAVED MODEL
# ----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fraud_detection_v1.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}. "
        "Run src/train.py first."
    )

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])
features = artifact["features"]

print("Fraud detection model loaded successfully!")


# ----------------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# ----------------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict fraudulent credit card transactions.",
    version="1.0.0"
)


# ----------------------------------------------------------
# 3. CREATE DYNAMIC PYDANTIC INPUT MODEL
# ----------------------------------------------------------

# Uses the exact feature names saved with the trained model.
# Every feature is required and must be numeric.

TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature: (float, ...)
        for feature in features
    }
)


# ----------------------------------------------------------
# 4. HEALTH CHECK ENDPOINT
# ----------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API is running!",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ----------------------------------------------------------
# 5. PREDICTION ENDPOINT
# ----------------------------------------------------------

@app.post("/predict")
def predict(transaction: TransactionInput):

    # Convert validated input into a DataFrame
    transaction_data = transaction.model_dump()

    input_df = pd.DataFrame(
        [transaction_data],
        columns=features
    )

    # Get fraud probability
    probability = float(
        pipeline.predict_proba(input_df)[0, 1]
    )

    # Apply saved decision threshold
    prediction = (
        probability >= threshold
    )

    result = (
        "Fraud" if prediction else "Legitimate"
    )

    # Return JSON response
    return {
        "fraud_probability": round(probability, 6),
        "threshold": threshold,
        "prediction": result
    }
