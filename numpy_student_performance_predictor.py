"""
Student Performance Predictor — NumPy from Scratch
=================================================
Requirements:
    pip install numpy matplotlib

Core ML operations use NumPy only.
Matplotlib is optional and used only for plots.

Features:
    1. Generate a synthetic dataset
    2. Explore data with NumPy statistics and correlations
    3. Filter records with Boolean masks
    4. Split into training and test sets
    5. Standardize features using training-set statistics
    6. Train multiple linear regression with gradient descent
    7. Evaluate with MAE, MSE, RMSE, and R²
    8. Predict a score for a new student

Note:
    This is synthetic educational data, not real student research data.
"""

import numpy as np


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

SEED = 42
N_STUDENTS = 500
TEST_FRACTION = 0.20
LEARNING_RATE = 0.05
EPOCHS = 2_000

FEATURE_NAMES = [
    "Study hours per day",
    "Attendance (%)",
    "Previous score",
    "Sleep hours per day",
]


# ------------------------------------------------------------
# 2. GENERATE A SYNTHETIC DATASET
# ------------------------------------------------------------

def generate_dataset(n_students=500, seed=42):
    """Return X (n_students, 4) and y (n_students,)."""
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0, 10, n_students)
    attendance = rng.uniform(40, 100, n_students)
    previous_score = rng.uniform(30, 100, n_students)
    sleep_hours = rng.uniform(4, 10, n_students)

    # Synthetic rule: each feature contributes to the final score.
    noise = rng.normal(0, 5, n_students)
    final_score = (
        3.0 * study_hours
        + 0.2 * attendance
        + 0.4 * previous_score
        + 1.5 * sleep_hours
        + noise
    )

    # Keep the target in the realistic score range.
    final_score = np.clip(final_score, 0, 100)

    # Shape: (n_students, 4). Each column is one feature.
    X = np.column_stack(
        (study_hours, attendance, previous_score, sleep_hours)
    )
    y = final_score
    return X, y


X, y = generate_dataset(N_STUDENTS, SEED)

print("=" * 65)
print("STUDENT PERFORMANCE PREDICTOR — NUMPY PROJECT")
print("=" * 65)
print(f"Input shape X: {X.shape}")
print(f"Target shape y: {y.shape}")
print("\nFirst 5 student rows (study, attendance, previous score, sleep):")
print(np.round(X[:5], 2))
print("\nFirst 5 final scores:")
print(np.round(y[:5], 2))


# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\nFeature statistics:")
print(f"{'Feature':<25} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")

for i, name in enumerate(FEATURE_NAMES):
    values = X[:, i]
    print(
        f"{name:<25} "
        f"{np.mean(values):>10.2f} "
        f"{np.median(values):>10.2f} "
        f"{np.std(values):>10.2f} "
        f"{np.min(values):>10.2f} "
        f"{np.max(values):>10.2f}"
    )

print("\nFinal-score statistics:")
print(f"Mean:   {np.mean(y):.2f}")
print(f"Median: {np.median(y):.2f}")
print(f"Std:    {np.std(y):.2f}")
print(f"Min:    {np.min(y):.2f}")
print(f"Max:    {np.max(y):.2f}")

# Correlation between each feature and the target score.
print("\nCorrelation of each feature with final score:")
correlations = np.array(
    [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
)

for name, corr in zip(FEATURE_NAMES, correlations):
    print(f"{name:<25}: {corr:+.4f}")

strongest_index = np.argmax(np.abs(correlations))
print(
    f"\nLargest absolute linear correlation: "
    f"{FEATURE_NAMES[strongest_index]} ({correlations[strongest_index]:+.4f})"
)

# A correlation is association, not proof of causation.
print("Reminder: correlation does not prove causation.")


# ------------------------------------------------------------
# 4. BOOLEAN MASKING / FILTERING
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 3 — BOOLEAN MASKING")
print("=" * 65)

study_col = X[:, 0]
attendance_col = X[:, 1]

studied_over_6 = study_col > 6
attendance_over_85 = attendance_col > 85
score_over_80 = y > 80
study_over_6_and_score_over_80 = (study_col > 6) & (y > 80)

print(f"Students studying more than 6 hours: {np.sum(studied_over_6)}")
print(f"Students with attendance above 85%: {np.sum(attendance_over_85)}")
print(f"Students scoring above 80: {np.sum(score_over_80)}")
print(
    "Students studying >6 hours AND scoring >80: "
    f"{np.sum(study_over_6_and_score_over_80)}"
)

# Show up to five matching rows.
matching_indices = np.where(study_over_6_and_score_over_80)[0]
print("\nFirst matching student indices:", matching_indices[:5])
print("Their input features:")
print(np.round(X[matching_indices[:5]], 2))
print("Their final scores:")
print(np.round(y[matching_indices[:5]], 2))


# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 4 — TRAIN / TEST SPLIT")
print("=" * 65)

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(X))

test_size = int(len(X) * TEST_FRACTION)
test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print("Training input shape:", X_train.shape)
print("Testing input shape: ", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape: ", y_test.shape)


# ------------------------------------------------------------
# 6. STANDARDIZE FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 5 — FEATURE STANDARDIZATION")
print("=" * 65)

# IMPORTANT: learn mean and std only from training data to avoid leakage.
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

# Protect against division by zero if a feature is constant.
train_std = np.where(train_std == 0, 1, train_std)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

print("Training means:", np.round(train_mean, 2))
print("Training stds: ", np.round(train_std, 2))
print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape: ", X_test_scaled.shape)


# ------------------------------------------------------------
# 7. LINEAR REGRESSION FROM SCRATCH
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 6 — TRAIN LINEAR REGRESSION WITH GRADIENT DESCENT")
print("=" * 65)

def train_linear_regression(X_data, y_data, learning_rate=0.05, epochs=2000):
    """
    Fit y_hat = X @ weights + bias by minimizing mean squared error.

    X_data shape: (m, n_features)
    y_data shape: (m,)
    weights shape: (n_features,)
    """
    m, n_features = X_data.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        # Forward pass: calculate predictions for every training row.
        predictions = X_data @ weights + bias

        # Error = prediction - actual target.
        errors = predictions - y_data

        # Gradients of mean squared error.
        dw = (2 / m) * (X_data.T @ errors)
        db = (2 / m) * np.sum(errors)

        # Update model parameters.
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Save loss occasionally, to avoid storing every epoch.
        if epoch % 100 == 0 or epoch == epochs - 1:
            updated_predictions = X_data @ weights + bias
            mse = np.mean((updated_predictions - y_data) ** 2)
            loss_history.append((epoch + 1, mse))

    return weights, bias, loss_history


weights, bias, loss_history = train_linear_regression(
    X_train_scaled,
    y_train,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
)

print("Learned weights (for standardized features):")
for name, weight in zip(FEATURE_NAMES, weights):
    print(f"{name:<25}: {weight:+.4f}")
print(f"Learned bias: {bias:.4f}")

print("\nTraining loss checkpoints (epoch, MSE):")
for epoch, loss in loss_history:
    print(f"Epoch {epoch:>4}: {loss:.4f}")


# ------------------------------------------------------------
# 8. MODEL EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 7 — EVALUATE ON UNSEEN TEST DATA")
print("=" * 65)

def predict(X_data, model_weights, model_bias):
    """Predict target values for already-standardized features."""
    return X_data @ model_weights + model_bias


def regression_metrics(y_true, y_pred):
    """Return MAE, MSE, RMSE, and R²."""
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return mae, mse, rmse, r2


y_train_pred = predict(X_train_scaled, weights, bias)
y_test_pred = predict(X_test_scaled, weights, bias)

train_mae, train_mse, train_rmse, train_r2 = regression_metrics(
    y_train, y_train_pred
)
test_mae, test_mse, test_rmse, test_r2 = regression_metrics(
    y_test, y_test_pred
)

print("TRAINING SET")
print(f"MAE:  {train_mae:.3f}")
print(f"MSE:  {train_mse:.3f}")
print(f"RMSE: {train_rmse:.3f}")
print(f"R²:   {train_r2:.3f}")

print("\nTEST SET")
print(f"MAE:  {test_mae:.3f}")
print(f"MSE:  {test_mse:.3f}")
print(f"RMSE: {test_rmse:.3f}")
print(f"R²:   {test_r2:.3f}")

print("\nFirst 10 test predictions:")
print(f"{'Actual':>10} {'Predicted':>12} {'Abs error':>12}")
for actual, predicted in zip(y_test[:10], y_test_pred[:10]):
    print(f"{actual:>10.2f} {predicted:>12.2f} {abs(actual-predicted):>12.2f}")


# ------------------------------------------------------------
# 9. PREDICT FOR A NEW STUDENT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODULE 8 — PREDICT FOR A NEW STUDENT")
print("=" * 65)

# Example student: 6 study hours, 90% attendance, previous score 75,
# and 7.5 hours of sleep per day.
new_student = np.array([[6.0, 90.0, 75.0, 7.5]])

# Use training mean/std, not statistics calculated from the new row.
new_student_scaled = (new_student - train_mean) / train_std
new_student_prediction = predict(new_student_scaled, weights, bias)[0]

# Regression can produce values outside 0–100; clip for this demo's score scale.
bounded_prediction = np.clip(new_student_prediction, 0, 100)

print("New student's features:")
for name, value in zip(FEATURE_NAMES, new_student[0]):
    print(f"{name:<25}: {value:.2f}")

print(f"\nRaw model prediction: {new_student_prediction:.2f}")
print(f"Prediction bounded to 0–100: {bounded_prediction:.2f}")


# ------------------------------------------------------------
# 10. OPTIONAL VISUALIZATION
# ------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    # Plot actual vs predicted test scores.
    plt.figure()
    plt.scatter(y_test, y_test_pred)
    plt.xlabel("Actual final score")
    plt.ylabel("Predicted final score")
    plt.title("Actual vs Predicted Scores (Test Set)")
    plt.plot([0, 100], [0, 100])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()

    # Plot training loss checkpoints.
    epochs_seen = np.array([item[0] for item in loss_history])
    losses_seen = np.array([item[1] for item in loss_history])

    plt.figure()
    plt.plot(epochs_seen, losses_seen)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Gradient Descent: Training Loss")
    plt.tight_layout()

    plt.show()

except ImportError:
    print("\nMatplotlib is not installed; skipping plots.")
    print("Optional install command: pip install matplotlib")


print("\nProject completed. Next: experiment with the learning rate,")
print("number of training examples, and input features.")
