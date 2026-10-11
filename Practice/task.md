#⚡ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score
)

#⚡ Step 2: Create Dataset
data = {
    "hours_studied":[2,4,1,5,3,6,7,8],
    "sleep_hours": [6,7,5,8,6,7,8,9],
    "attendance": [60,80,50,90,70,85,95,98],
    "marks": [40,60,30,70,50,75,85,92]
}

df = pd.DataFrame(data)

# ⚡ Step 3: Data Cleaning
# 🔹 Add Missing Value
df.loc[2,"marks"] = None

#🔹 Handle Missing Value
df["marks"].fillna(df["marks"].mean(), inplace=True)
print(df.isnull().sum())

# ⚡ Step 4: Feature Engineering
# 🔹 Create New Features
df["total_effort"] = (
    df["hours_studied"] + df["attendance"]/10
)

df["study_sleep_ratio"] = (
    df["hours_studied"] / df["sleep_hours"]
)

# ⚡ Step 5: Create Pass/Fail Column
df["status"] = df["marks"].apply(
    lambda x: 1 if x >= 50 else 0
)

# ⚡ Step 6: Visualization
# 📊 Scatter Plot

plt.scatter(df["hours_studied"], df["marks"])
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Study vs Marks")
plt.show()

# ⚡ Step 7: Prepare Features
X = df[[
    "hours_studied",
    "sleep_hours",
    "attendance",
    "total_effort",
    "study_sleep_ratio"
]]

y_marks = df["marks"]
y_status = df["status"]

# ⚡ Step 8: Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ⚡ Step 9: Train/Test Split
valid_rows = y_marks.notna().to_numpy()
X_valid = X_scaled[valid_rows]
y_valid = y_marks[valid_rows].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_valid,
    y_valid,
    test_size=0.2,
    random_state=42
)

# 🚀 Step 10: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Linear Regression Error:")
print(mean_absolute_error(y_test, lr_pred))


# 🌳 Step 11: Decision Tree
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print("Decision Tree Error:")
print(mean_absolute_error(y_test, tree_pred))

# 🤝 Step 12: KNN Model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print("KNN Error:")
print(mean_absolute_error(y_test, knn_pred))

# ⚡ Step 14: Pass/Fail Classifier
clf_model = LogisticRegression()

clf_model.fit(X_scaled, y_status)

new_student = scaler.transform([
    [5,7,80,13,0.7]
])

result = clf_model.predict(new_student)

if result[0] == 1:
    print("Pass ✅")
else:
    print("Fail ❌")
    
# 🔥 Step 15: Final Prediction System
hours = float(input("Hours Studied: "))
sleep = float(input("Sleep Hours: "))
attendance = float(input("Attendance: "))

total_effort = hours + attendance/10
ratio = hours / sleep

new_data = scaler.transform([
    [hours, sleep, attendance, total_effort, ratio]
])

pred_marks = lr_model.predict(new_data)

print("\nPredicted Marks:", round(pred_marks[0],2))

print("Linear:", lr_model.score(X_train, y_train))
print("Tree:", tree_model.score(X_train, y_train))
print("KNN:", knn_model.score(X_train, y_train))