# Part2_Iris_DecisionTree.py (or use in Jupyter cells)
"""
Practical implementation:
- Load Iris dataset
- Preprocess (handle missing values, encode labels)
- Train Decision Tree classifier
- Evaluate using accuracy, precision, recall
- Well-commented for clarity
"""

# -------------------------
# 1) Imports
# -------------------------
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Optional: to visualize tree (commented out by default)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# -------------------------
# 2) Load dataset
# -------------------------
# Using scikit-learn's built-in Iris dataset
iris = load_iris(as_frame=True)  # as_frame=True returns pandas DataFrame
X = iris.data.copy()             # features as DataFrame
y = iris.target.copy()           # numeric labels (0,1,2)
feature_names = iris.feature_names
target_names = iris.target_names

# For clarity, convert to a single DataFrame (optional but helpful)
df = X.copy()
df['species'] = y

# Show first few rows (in a notebook you can display df.head())
print("First 5 rows of the dataset:")
print(df.head())

# -------------------------
# 3) Simulate / handle missing values (if dataset had any)
# -------------------------
# The real iris dataset has no missing values. 
# Still, we show how to handle missing values so the pipeline is complete.

# Example: (commented) To simulate missing data for demonstration uncomment:
# rng = np.random.default_rng(42)
# df_missing = df.copy()
# mask = rng.choice([True, False], size=df_missing.shape, p=[0.05, 0.95])  # ~5% missing
# df_missing[df_missing.columns] = df_missing.where(~mask, other=np.nan)
# df = df_missing
# X = df.drop(columns=['species'])
# y = df['species']

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# If there were missing numeric values, use SimpleImputer to fill them
# We'll demonstrate standard numeric imputation (mean strategy).
imputer = SimpleImputer(strategy='mean')  # numeric features -> replace missing with mean

# Fit the imputer on feature columns and transform
X_imputed = pd.DataFrame(imputer.fit_transform(df[feature_names]), columns=feature_names)

# -------------------------
# 4) Encode labels
# -------------------------
# In this dataset, labels are already numeric (0,1,2). If we had string labels, we would:
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(df['species'])
# For clarity use y as-is (numeric)
y_encoded = df['species'].values

# If you prefer to show mapping:
label_map = {i: name for i, name in enumerate(target_names)}
print("\nLabel mapping (numeric -> species):")
print(label_map)

# -------------------------
# 5) Train-test split
# -------------------------
# Split into training and test sets. Use stratify=y to preserve class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -------------------------
# 6) Train Decision Tree Classifier
# -------------------------
# Instantiate the classifier. You can tune max_depth, min_samples_leaf, criterion, etc.
clf = DecisionTreeClassifier(random_state=42)

# Fit model on training data
clf.fit(X_train, y_train)

# -------------------------
# 7) Predictions
# -------------------------
y_pred = clf.predict(X_test)

# -------------------------
# 8) Evaluation
# -------------------------
# Accuracy
acc = accuracy_score(y_test, y_pred)

# Precision and recall:
# For multi-class, specify average. 'macro' = unweighted mean, 'weighted' accounts for class support.
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall    (macro): {recall_macro:.4f}")
print(f"Precision (weighted): {precision_weighted:.4f}")
print(f"Recall    (weighted): {recall_weighted:.4f}")

# Full classification report (precision, recall, f1-score per class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print("Confusion Matrix:")
print(cm_df)

