# train_models.py

import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")

# =====================================================
# CREATE DIRECTORIES
# =====================================================

os.makedirs("models", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

# Replace with your dataset path
df = pd.read_csv("adult.csv")

print("Dataset Shape:", df.shape)

# =====================================================
# DATA CLEANING
# =====================================================

df = df.drop_duplicates()

# Replace missing values represented by ?
df.replace("?", np.nan, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# =====================================================
# TARGET COLUMN
# =====================================================

target_column = "income"

# Convert target to 0/1

df[target_column] = df[target_column].apply(
    lambda x: 1 if ">50K" in str(x) else 0
)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop(columns=[target_column])
y = df[target_column]

# =====================================================
# IDENTIFY COLUMN TYPES
# =====================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)

# =====================================================
# PREPROCESSOR
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# SAVE TEST DATA FOR STREAMLIT
# =====================================================

test_data = X_test.copy()
test_data[target_column] = y_test

test_data.to_csv(
    "test_data.csv",
    index=False
)

print("test_data.csv saved")

# =====================================================
# PREPROCESS DATA
# =====================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

# =====================================================
# MODELS
# =====================================================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB(),

    "Support Vector Machine":
        SVC(
            probability=True,
            random_state=42
        )
}

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(
    model,
    x_test,
    y_test
):

    y_pred = model.predict(x_test)

    try:
        y_prob = model.predict_proba(x_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    except:
        auc = np.nan

    acc = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }

# =====================================================
# TRAIN MODELS
# =====================================================

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    # ----------------------------------
    # Naive Bayes handling
    # ----------------------------------

    if model_name == "Naive Bayes":

        model.fit(
            X_train_processed.toarray(),
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test_processed.toarray(),
            y_test
        )

    else:

        model.fit(
            X_train_processed,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test_processed,
            y_test
        )

    # ----------------------------------
    # SAVE MODEL
    # ----------------------------------

    file_name = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        f"models/{file_name}"
    )

    print(
        f"Saved: models/{file_name}"
    )

    row = {
        "Model": model_name
    }

    row.update(metrics)

    results.append(row)

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
]

print("\n")
print("=" * 80)
print(results_df)
print("=" * 80)

# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print(
    "\nmodel_comparison_results.csv saved"
)

print(
    "All models trained successfully."
)