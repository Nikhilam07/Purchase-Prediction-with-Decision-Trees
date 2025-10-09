# Purchase-Prediction-with-Decision-Trees
# Project 2 — Purchase Prediction with Decision Trees



---

## 1. Title

**Purchase Prediction with Decision Trees**

## 2. Abstract

This project builds a Decision Tree classifier to predict whether a customer will make a purchase based on features such as Age, Gender, and Estimated Salary. We use a common public dataset (often named `Social_Network_Ads.csv`), perform preprocessing, train a Decision Tree, evaluate its performance, and visualize the model.

## 3. Objective

* Build a machine learning pipeline that reads the dataset, preprocesses features, trains a Decision Tree classifier, and evaluates model performance.
* Provide a clean, reproducible Python implementation suitable for submission.

## 4. Dataset Description

We use a dataset with the following columns (common variant `Social_Network_Ads.csv`):

* `User ID` (optional)
* `Gender` — categorical (`Male`/`Female`)
* `Age` — numeric
* `EstimatedSalary` — numeric
* `Purchased` — target (0 or 1)

> **Note:** Place the `Social_Network_Ads.csv` file in the same folder as the code, or change the path accordingly.

## 5. Data Preprocessing

Steps performed in the pipeline:

1. Load data with pandas.
2. Drop `User ID` if present.
3. Encode `Gender` to numeric (0/1) using `LabelEncoder` or `pd.get_dummies`.
4. Separate features (`X`) and target (`y`).
5. Split into train and test sets using `train_test_split` (test_size=0.25, random_state=42).
6. Scale numeric features (`Age`, `EstimatedSalary`) using `StandardScaler` (scaling improves numeric stability for many models; Decision Trees don't strictly require scaling but we demonstrate it for pipeline consistency).

## 6. Model Building — Decision Tree

* Use `sklearn.tree.DecisionTreeClassifier`.
* Use `criterion='entropy'` or `'gini'`. `entropy` tends to give slightly different splits; both are reasonable.
* Tune hyperparameters like `max_depth`, `min_samples_split` optionally via `GridSearchCV`.

## 7. Implementation (Python code)

Below is a complete, runnable Python script. Save it as `purchase_decision_tree.py` and place the dataset `Social_Network_Ads.csv` in the same folder.

```python
# purchase_decision_tree.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# 1. Load the data
DATA_PATH = 'Social_Network_Ads.csv'  # change path if needed

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place Social_Network_Ads.csv in the working directory.")

# 2. Inspect
print('First 5 rows:')
print(df.head())
print('\nDataset info:')
print(df.info())
print('\nClass distribution:')
print(df['Purchased'].value_counts())

# 3. Preprocessing
# Drop User ID if present
if 'User ID' in df.columns:
    df = df.drop(columns=['User ID'])

# Encode Gender
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Male/Female -> 1/0

# Features and target
feature_cols = [c for c in df.columns if c != 'Purchased']
X = df[feature_cols].values
y = df['Purchased'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Scale numeric columns (Age, EstimatedSalary). We will scale all numeric columns in X.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model training: Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}")
print('\nClassification report:')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix:')
print(cm)

# 6. Visualizations
# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5,4))
ax.matshow(cm, cmap='Blues', alpha=0.7)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), va='center', ha='center')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.show()

# Plot the decision tree (may be large for deep trees)
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=feature_cols, class_names=['Not Purchased','Purchased'], filled=True, rounded=True)
plt.title('Decision Tree')
plt.show()

# 7. Save model and scalers
joblib.dump(clf, 'decision_tree_purchase_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('\nModel and scaler saved as joblib files.')

# 8. (Optional) Hyperparameter tuning with GridSearchCV
# Uncomment to run
# param_grid = {
#     'max_depth': [2, 3, 4, 5, 6, None],
#     'min_samples_split': [2, 5, 10],
#     'criterion': ['gini', 'entropy']
# }
# grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
# grid.fit(X_train, y_train)
# print('Best params:', grid.best_params_)
# print('Best CV score:', grid.best_score_)

```

## 8. Results and Evaluation

* **Primary metric:** Accuracy on the test set.
* **Additional metrics:** Precision, recall, F1-score per class (see classification report).
* **Interpretation:** Decision Trees are interpretable — use `plot_tree` to understand splitting rules.

> Example typical results (will vary by dataset and split):
>
> * Test accuracy: `~0.90` (example)
> * Class balance important: if dataset is skewed, consider using balanced class weights or resampling.

## 9. Discussion & Improvements

* Decision Trees can overfit if not pruned. Use `max_depth`, `min_samples_leaf` to limit complexity.
* Test Grid Search or Randomized Search for hyperparameter tuning.
* Try ensemble methods: RandomForest, GradientBoosting, or XGBoost for better performance and robustness.
* For unbalanced data, try SMOTE for oversampling or `class_weight='balanced'` in the classifier.
* Perform feature engineering: interaction terms, binning Age, or using polynomial features if needed.

## 10. Deliverables

* `purchase_decision_tree.py` — runnable script (above).
* `decision_tree_purchase_model.joblib` — saved trained model (created after running script).
* `scaler.joblib` — saved scaler.

## 11. How to run

1. Ensure Python 3.8+ is installed.
2. Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

3. Place `Social_Network_Ads.csv` in the same folder.
4. Run:

```bash
python purchase_decision_tree.py
```

---

## Appendix: Short README (copy for submission)

**Project:** Purchase Prediction with Decision Trees
**Description:** Predict if a customer will purchase using a Decision Tree classifier trained on basic demographic and salary features.
**Files:** `Social_Network_Ads.csv`, `purchase_decision_tree.py`
**How to run:** see section "How to run" above.

---

*End of document.*

