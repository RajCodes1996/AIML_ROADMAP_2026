#### Day 17: In progress / unconfirmed

GridSearchCV, hyperparameter tuning and cross-validation. The lesson and code were delivered, but your execution and results haven't been confirmed. Hyperparameter tuning and cross-validation — still to verify

. What we should reinforce
The handover identifies four areas worth revisiting during practical projects:
- EDA: understanding univariate versus multivariate analysis.
- Model evaluation: knowing when accuracy is misleading and when precision, recall or F1-score is more useful.
- Generalization: distinguishing training, validation and test data.
- Model understanding: explaining why the code and model work, not just running them.
These are reinforcement priorities, not evidence that you have failed to learn those concepts. ML_Roadmap_2026_Handover (1).md

4. Let's finish Day 17 first


Day 17 — Hyperparameter Tuning
Project: Optimize a Random Forest classifier using the Breast Cancer Wisconsin dataset.
Tools: Python, Pandas and scikit-learn



Day 17 — Results checklist
Copy the values from your VS Code or Colab output. If you don't have a value handy, leave it blank.
1. Baseline test accuracy

2. Best parameters

3. Best cross-validation accuracy

4. Tuned test accuracy

5. Classification report (optional)

6. Top five search results (optional)

Step 2: Understand what we're checking
Comparison	What it tells us
Baseline vs tuned test accuracy	Whether the tuned model improved on this test set
best_score_	Best mean cross-validation score found during the search
Precision, recall and F1-score	How well the model handles each class
Top five search results	Whether several parameter combinations perform similarly

A tuned model does not necessarily outperform the baseline. Similar scores can still be a useful result.
Step 3: One important concept question
Why do we keep the test set separate when GridSearchCV already performs cross-validation?