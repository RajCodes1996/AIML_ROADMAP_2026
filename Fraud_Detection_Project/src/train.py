
# ==========================================================
# CREDIT CARD FRAUD DETECTION - DAY 21
# End-to-End Machine Learning Project V1
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ----------------------------------------------------------
# 1. CREATE PROJECT DIRECTORIES
# ----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------

print("\nLoading Credit Card Fraud Dataset...")

dataset = fetch_openml(
    data_id=1597,
    as_frame=True
)

df = dataset.frame

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ----------------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# ----------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ----------------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT
# ----------------------------------------------------------

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print("\nData Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ----------------------------------------------------------
# 5. BUILD MACHINE LEARNING PIPELINE
# ----------------------------------------------------------

fraud_pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# ----------------------------------------------------------
# 6. TRAIN MODEL
# ----------------------------------------------------------

print("\nTraining Model...")

fraud_pipeline.fit(X_train, y_train)

print("Model Training Completed!")


# ----------------------------------------------------------
# 7. VALIDATION EVALUATION
# ----------------------------------------------------------

val_predictions = fraud_pipeline.predict(X_val)

print("\nVALIDATION RESULTS")

print("Accuracy:", accuracy_score(y_val, val_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

print("\nClassification Report:")
print(classification_report(
    y_val,
    val_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))


# ----------------------------------------------------------
# 8. THRESHOLD SELECTION USING VALIDATION DATA
# ----------------------------------------------------------

val_probabilities = fraud_pipeline.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

eligible = threshold_df[
    threshold_df["Recall"] >= 0.80
]

if eligible.empty:
    raise RuntimeError(
        "No tested threshold achieved 80% validation recall."
    )

best_row = eligible.sort_values(
    "Precision",
    ascending=False
).iloc[0]

chosen_threshold = float(best_row["Threshold"])

print("\nSELECTED THRESHOLD")
print("Threshold:", chosen_threshold)
print("Validation Precision:", best_row["Precision"])
print("Validation Recall:", best_row["Recall"])
print("Validation F1:", best_row["F1"])


# ----------------------------------------------------------
# 9. FINAL TEST EVALUATION - RUN ONCE
# ----------------------------------------------------------

test_probabilities = fraud_pipeline.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_matrix = confusion_matrix(y_test, test_predictions)

test_report = classification_report(
    y_test,
    test_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
)

print("\nFINAL TEST CONFUSION MATRIX")
print(test_matrix)

print("\nFINAL TEST CLASSIFICATION REPORT")
print(test_report)


# ----------------------------------------------------------
# 10. SAVE TRAINED MODEL AND THRESHOLD
# ----------------------------------------------------------

model_artifact = {
    "pipeline": fraud_pipeline,
    "threshold": chosen_threshold,
    "features": list(X.columns)
}

model_path = os.path.join(
    MODEL_DIR,
    "fraud_detection_v1.joblib"
)

joblib.dump(model_artifact, model_path)

print("\nModel Saved Successfully!")
print("Location:", model_path)


# ----------------------------------------------------------
# 11. SAVE EVALUATION REPORT
# ----------------------------------------------------------

report_path = os.path.join(
    REPORT_DIR,
    "evaluation.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("CREDIT CARD FRAUD DETECTION - V1\n\n")

    file.write(f"Selected Threshold: {chosen_threshold}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(test_matrix))

    file.write("\n\nClassification Report:\n")
    file.write(test_report)

print("Evaluation Report Saved!")
print("Location:", report_path)


# ----------------------------------------------------------
# 12. TEST MODEL LOADING AND PREDICTION
# ----------------------------------------------------------

loaded_artifact = joblib.load(model_path)

loaded_pipeline = loaded_artifact["pipeline"]
loaded_threshold = loaded_artifact["threshold"]

sample = X_test.iloc[[0]]

probability = loaded_pipeline.predict_proba(sample)[0, 1]

prediction = int(probability >= loaded_threshold)

print("\nSAMPLE PREDICTION")
print("Fraud Probability:", probability)
print(
    "Prediction:",
    "Fraud" if prediction == 1 else "Legitimate"
)

print("\nDAY 21 PROJECT EXECUTION COMPLETED!")


# Expected project outputs
# Dataset loaded

# Model trained

# Validation threshold selected

# Final test report generated

# Joblib model saved

# Evaluation report saved
