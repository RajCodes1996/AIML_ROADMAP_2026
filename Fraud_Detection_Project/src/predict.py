
# ==================================================
# DAY 22 - LOAD SAVED MODEL AND MAKE PREDICTIONS
# ==================================================

import os
import joblib
import pandas as pd

# 1. Locate project directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 2. Load saved model artifact
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fraud_detection_v1.joblib"
)

print("Loading saved model...")
artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
features = artifact["features"]

print("Model loaded successfully!")
print("Model threshold:", threshold)

# 3. Create sample transaction data
# Values here are illustrative, not real transaction records.

sample_data = {
    feature: [0.0] for feature in features
}

sample_data["Amount"] = [100.0]
sample_data["Time"] = [1000.0]
new_transaction = pd.DataFrame(sample_data)

# Ensure correct feature order
new_transaction = new_transaction[features]

# 4. Make prediction
fraud_probability = pipeline.predict_proba(
    new_transaction
)[0,1]

prediction = int(
    fraud_probability >= threshold
)

# Display Results
print("\nTRANSACTION ANALYSIS")
print("--------------------")

print(
    f"Fraud Probability: {fraud_probability:.4f}"
)
print(
    f"Decision Threshold: {threshold:.2%}"
)

if prediction == 1:
    print("RESULT: FRAUD DETECTED")
else:
    print("RESULT: LEGITIMATE TRANSACTION")