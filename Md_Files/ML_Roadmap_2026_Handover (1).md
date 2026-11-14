# ML Roadmap 2026 — 30-Day Learning Plan Handover

**Prepared:** 22 October 2026  
**Purpose:** Self-contained context for continuing in a new ChatGPT conversation.

## Current status
- **Confirmed complete:** Days 1–16.
- **Current lesson:** Day 17 — Hyperparameter Tuning with GridSearchCV and cross-validation. Lesson/code delivered, but completion has **not** been explicitly confirmed.
- **Immediate next action:** Run/finish Day 17; record baseline test accuracy, `best_params_`, `best_score_`, tuned test accuracy, classification report, and top five search results. Answer why the test set remains separate.
- **Next lesson after Day 17:** Day 18 — Real Datasets and a fraud-detection workflow.

> Source note: the earliest roadmap's complete verbatim text was not retrievable. This is a careful reconstruction from the user's prior messages and the October 2026 refreshed roadmap. Confirmed completion is separated from planned content.

## User goals and learning preferences
- User describes themself as an AIML engineer / professional Python developer and wants to strengthen an ML portfolio with current, practical technologies.
- Long-term goal: become an Assistant Professor in Computer Science and AIML by 2028.
- Prefers project-first, productive learning rather than traditional theory-only learning.
- Wants daily steps, simple explanations, runnable Python code for Google Colab or VS Code, small practical exercises, checklists, and explicit “DAY X DONE” milestones.
- Wants concise code when requested, but needs concepts explained clearly.
- Previously said univariate and multivariate EDA were confusing; revisit with simple examples if useful.
- Reinforce test vs validation data, overfitting/generalization, and why accuracy alone can mislead on imbalanced data.
- Do not infer model scores or claim a tuned model improved unless the user provides outputs.

## Strengths and areas to reinforce
**Observed strengths:** consistent progress and milestone reporting; learns by building; has practiced multiple classical model families; interested in evaluation and portfolio projects.

**Reinforce:** EDA basics if needed; train/validation/test separation; accuracy vs precision/recall/F1; overfitting; ability to explain the code, not just run it. These are teaching priorities, not a diagnosis of ability.

## Reconstructed 30-day roadmap

| Day | Topic | Status |
|---|---|---|
| 1 | Setup, inspect first dataset, missing values, scatter plot | Complete |
| 2 | NumPy/Pandas, filtering, new columns, sorting, missing values, grouping; Student Analyzer | Complete |
| 3 | Visualization, correlation, outlier challenge, EDA | Complete |
| 4 | Linear Regression, prediction, fitted line, slope/intercept, user input | Complete |
| 5 | Train/test split, MAE, validation basics | Complete |
| 6 | Logistic Regression pass/fail classification, accuracy | Complete |
| 7 | Student Performance Predictor V1: regression + classification | Complete |
| 8 | Cleaning: missing values, duplicates, numeric conversion | Complete |
| 9 | Feature engineering, derived features, scaling | Complete |
| 10 | Decision Tree Regressor, tree visualization, depth/overfitting | Complete |
| 11 | KNN Regressor, scaling, model comparison | Complete |
| 12 | Confusion matrix, precision, recall, classification report; class imbalance | Complete |
| 13 | Overfitting/underfitting; train/test score comparison, tree depths 1/3/5 | Complete |
| 14 | Student Performance Predictor V2: cleaning, engineered features, multiple models, evaluation, custom input | Complete |
| 15 | Random Forest vs Decision Tree regression | Complete |
| 16 | Ensemble Learning and Voting Classifier | Complete |
| 17 | GridSearchCV, hyperparameter tuning, cross-validation | In progress / not confirmed |
| 18 | Real dataset, fraud detection exploration and baseline model | Pending |
| 19 | Imbalanced classification, precision/recall/F1, thresholds, class weights | Pending |
| 20 | `Pipeline`, `ColumnTransformer`, preprocessing, leakage prevention | Pending |
| 21 | End-to-end Fraud Detection Project V1 | Pending |
| 22 | Save/load models with `joblib`, inference and version awareness | Pending |
| 23 | FastAPI prediction API, Pydantic validation | Pending |
| 24 | Testing, error handling, Docker/deployment fundamentals | Pending |
| 25 | PyTorch tensors, tensor operations, autograd, training workflow | Pending |
| 26 | ANN: layers, activations, loss, optimizer, backprop, training loop | Pending |
| 27 | Pretrained NLP models, tokenization, Hugging Face Transformers | Pending |
| 28 | Embeddings, semantic search, vector stores, RAG | Pending |
| 29 | Integrate selected components into a final project | Pending |
| 30 | Portfolio review, README, GitHub, results, next roadmap | Pending |

## Completed topics and important explanations

### Days 1–7: Foundation
- Day 1: setup, first dataset, inspect data, missing-value handling, scatter plot.
- Day 2: NumPy/Pandas, filtering, derived columns, sorting, grouping; Student Analyzer.
- Day 3: scatter/line/bar/histogram plots, correlation, outliers. Earlier confusion around univariate vs multivariate analysis was noted.
- Day 4: Linear Regression, predictions, fitted line, slope/intercept, user input.
- Day 5: train/test split, MAE, validation basics.
- Day 6: Logistic Regression for pass/fail classification and accuracy.
- Day 7: Student Performance Predictor V1 using regression and classification.

### Days 8–14: Classical ML and Student Predictor V2
- Day 8: missing values, duplicates, numeric conversion.
- Day 9: feature engineering, derived features, scaling. Scaling is particularly important for distance-based KNN.
- Day 10: Decision Tree Regressor, depth, visualization, overfitting.
- Day 11: KNN regression, scaling, comparison.
- Day 12: confusion matrix, precision, recall, classification report. Challenge: imbalanced Pass/Fail labels and comparing accuracy vs precision/recall. False positives and false negatives matter.
- Day 13: compare tree depths 1, 3, 5 and train/test scores. High training performance with much lower test performance indicates overfitting.
- Day 14: Student Performance Predictor V2 used `hours_studied`, `sleep_hours`, `attendance`, engineered features, data cleaning, Linear Regression, Decision Tree, KNN, Logistic Regression, evaluation, and custom input. User confirmed project V2 complete.

### Day 15: Random Forest vs Decision Tree
- Used a small synthetic student regression dataset with `hours_studied`, `sleep_hours`, `attendance`, and `marks`.
- Compared `DecisionTreeRegressor(max_depth=3, random_state=42)` with `RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)`.
- Evaluated train/test MAE and test R²; made a custom prediction.
- Random Forest aggregates many trees (averaging for regression). It often generalizes better but is not guaranteed to beat one tree, especially on tiny synthetic data.
- User explicitly said **DAY 15 DONE**.

Core comparison pattern:
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=42
    )
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", mean_absolute_error(y_test, pred),
          "R2:", r2_score(y_test, pred))
```

### Day 16: Ensemble Learning and Voting Classifier
- Bagging: train models on bootstrap samples and aggregate; Random Forest is a bagging-based tree ensemble.
- Voting: hard voting combines class votes; soft voting averages class probabilities.
- Boosting: sequentially builds models to improve on earlier errors/residuals; examples AdaBoost, Gradient Boosting, XGBoost.
- Mini-project: compare Decision Tree, Random Forest, Logistic Regression, and VotingClassifier on a small synthetic pass/fail dataset (`hours_studied`, `attendance`, `passed`); evaluate accuracy and `classification_report`; predict for a new student.
- User explicitly said **DAY 16 DONE**.

Core VotingClassifier pattern:
```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

ensemble = VotingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000))
    ],
    voting="hard"
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```
Try `voting="soft"` as an extension; all component models must support `predict_proba`.

## Day 17 — Hyperparameter Tuning with GridSearchCV (current lesson)

**Dataset:** scikit-learn Breast Cancer Wisconsin dataset, malignant vs benign classification.  
**Split:** 80% train / 20% test, `random_state=42`, `stratify=y`.  
**Baseline:** `RandomForestClassifier(random_state=42)`.  
**Grid:** `n_estimators=[50,100]`, `max_depth=[None,5,10]`, `min_samples_split=[2,5]`, `min_samples_leaf=[1,2]`. This is 24 combinations; with `cv=5`, 120 CV fits.  
**Scoring:** accuracy; `n_jobs=-1`, `verbose=1`.

Complete runnable code:
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, baseline_predictions))

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
tuned_predictions = best_model.predict(X_test)
print("Tuned accuracy:", accuracy_score(y_test, tuned_predictions))
print(classification_report(
    y_test, tuned_predictions, target_names=data.target_names
))

results = pd.DataFrame(grid_search.cv_results_)
top_results = results[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score").head(5)
print(top_results.to_string(index=False))
```

**Critical explanation:** GridSearchCV tunes using cross-validation splits within the training set. The test set is held untouched until the end, to provide a more independent final estimate of generalization. Do not use test scores to keep choosing hyperparameters.

**Checklist:** record baseline accuracy; run GridSearchCV; record `best_params_` and `best_score_`; compare baseline and tuned test accuracy; inspect top five results; answer why the test set is kept separate; optionally change the grid and rerun.

**Completion status:** lesson/code delivered, but no explicit `DAY 17 DONE` confirmation is in the record. Do not assume the results or that tuning improved accuracy.

## Pending days 18–30
- **18:** Real dataset workflow and fraud-detection baseline.
- **19:** Class imbalance, precision/recall/F1, thresholds, class weighting; use suitable metrics.
- **20:** `Pipeline`, `ColumnTransformer`, numeric/categorical preprocessing, leakage prevention.
- **21:** End-to-end Fraud Detection V1.
- **22:** Save/load model using `joblib`; inference script and version awareness.
- **23:** FastAPI prediction API, Pydantic validation.
- **24:** Test model/API, error handling, Docker and deployment fundamentals.
- **25:** PyTorch tensors, autograd, training workflow.
- **26:** ANN, activations, losses, optimizers, backpropagation, training loop.
- **27:** Tokenization and pretrained Hugging Face Transformers inference.
- **28:** Embeddings, semantic search, vector store, RAG, grounded retrieval.
- **29:** Integrate selected components into a working project.
- **30:** Portfolio polish: README, GitHub, setup, metrics, limitations, demo, next plan.

## Important concept cheat sheet
- **Regression vs classification:** numeric value vs category.
- **MAE:** mean absolute numeric prediction error; lower is better on the same target scale.
- **Accuracy:** overall fraction correct; can hide poor minority-class recall.
- **Precision:** among predicted positives, fraction truly positive.
- **Recall:** among actual positives, fraction found by the model.
- **Confusion matrix:** counts TP, TN, FP, FN.
- **Overfitting:** strong training fit but weaker unseen-data performance.
- **Feature engineering:** transform/create useful inputs.
- **Scaling:** put numeric inputs on comparable scales; especially important for KNN.
- **Random Forest:** combines trees; average regression outputs or aggregate class outputs.
- **Bagging / voting / boosting:** bootstrap aggregation / combine classifier votes or probabilities / sequential improvement.
- **Hyperparameter:** model setting chosen before fitting.
- **GridSearchCV:** searches a specified parameter grid using CV and a scoring metric.
- **Cross-validation:** repeated training/validation folds to estimate candidate performance.
- **Data leakage:** training or model selection improperly uses information from validation/test or unavailable-at-prediction-time information.
- **Held-out test set:** untouched during tuning; used for final evaluation.

## Earlier exercises and topics
- EDA practice CSV: 20 students with `Sex = Male/Female`, `Smoker = Yes/No`, with Colab visualizations.
- Final revision project after framing a problem, CSV, JSON/SQL, API fetching, and web scraping; user requested full code.
- Day 12: imbalanced pass/fail labels, accuracy vs precision/recall, false positives/false negatives.
- Day 13: decision tree depths 1/3/5 and train/test score comparison.
- Day 14: Student Performance Predictor V2.
- Day 15: Random Forest vs Decision Tree regression.
- Day 16: VotingClassifier comparison.
- Day 17: GridSearchCV on Breast Cancer Wisconsin data.

## Open checks
1. Day 17 completion is not confirmed.
2. Quiz not answered in the recorded exchange: **Why retain a separate test set when using GridSearchCV?** Expected: CV supports tuning on training data; the untouched test set gives a final independent estimate.
3. No explicit errors or doubts reported for Days 15–16.
4. Earlier EDA confusion may merit a refresher.
5. Exact verbatim original roadmap text was not recovered; this plan is reconstructed from conversation history and October 2026 refresh.

## Instructions for the next assistant
1. Acknowledge Days 1–16 are complete and Day 17 is not confirmed.
2. Do not restart at Day 1.
3. Help finish/debug Day 17 and ask for actual baseline accuracy, `best_params_`, `best_score_`, and tuned test accuracy if available.
4. Once user confirms `DAY 17 DONE`, begin **Day 18: Real Datasets and Fraud Detection** with runnable code, data inspection, cleaning, baseline model, appropriate metrics, checklist, and completion prompt.
5. Keep the lessons practical, structured, simple, and honest about the limitations of small synthetic datasets.

## Copy/paste prompt for a new chat
Continue my ML Roadmap 2026 using this handover as the source of truth. I have explicitly completed Days 1–16. Day 17 (GridSearchCV and five-fold cross-validation on scikit-learn's Breast Cancer Wisconsin dataset) has been taught, but I have not yet explicitly confirmed completion. First help me finish or verify Day 17: baseline RandomForestClassifier, 24 hyperparameter combinations, cv=5 (120 CV fits), best_params_, best_score_, tuned model test accuracy, classification report, and top five cv_results_. Explain why the held-out test set must remain separate from tuning. Once I say DAY 17 DONE, begin Day 18: Real Datasets and a fraud-detection workflow. Teach me in a practical, step-by-step way with runnable Python code, simple explanations, exercises, and a checklist. Do not restart the roadmap from Day 1.
