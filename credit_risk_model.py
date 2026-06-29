"""
Give Me Some Credit - Creditworthiness Prediction

Target column : SeriousDlqin2yrs 
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)

RANDOM_STATE = 42


# 1. LOAD DATA

TRAIN_PATH = "cs-training.csv"   
TEST_PATH = "cs-test.csv"        

df = pd.read_csv(TRAIN_PATH, index_col=0)
print("Raw shape:", df.shape)
print(df["SeriousDlqin2yrs"].value_counts(normalize=True))

# 2. FEATURE ENGINEERING

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    #  Handle missing values 
    # MonthlyIncome: fill with median (income is right-skewed)
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())
    # NumberOfDependents:
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(0)

    #  Fix known data issues
    # age has a few rows with age=0 -> treat as missing, impute with median
    data.loc[data["age"] < 18, "age"] = data["age"].median()

    # Past-due columns have placeholder outlier values (96, 98) 
    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"]:
        data[col] = data[col].clip(upper=data[col].quantile(0.999))

    #  New engineered features
    data["TotalPastDue"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )

    data["DebtToIncome"] = data["DebtRatio"] * data["MonthlyIncome"]
    data["IncomePerDependent"] = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)
    data["CreditLinesPerAge"] = data["NumberOfOpenCreditLinesAndLoans"] / data["age"]
    data["HasRealEstateLoan"] = (data["NumberRealEstateLoansOrLines"] > 0).astype(int)
    data["UtilizationBucket"] = pd.cut(
        data["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.01, 0.3, 0.7, 1.0, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    data["IsRetired"] = (data["age"] >= 65).astype(int)

    # Remve extreme outliers in ratio-type columns (winsorize at 99th pct)
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = data[col].quantile(0.99)
        data[col] = data[col].clip(upper=cap)

    return data


df = engineer_features(df)
print("\nAfter feature engineering, shape:", df.shape)
print(df.head())

# 3. TRAIN / TEST SPLIT

y = df["SeriousDlqin2yrs"]
X = df.drop(columns=["SeriousDlqin2yrs"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Scale features (mainly helps Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=50, class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")

    # Logistic Regression benefits from scaled data; tree models don't need it
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}  |  F1-Score: {f1:.4f}")

    results[name] = {
        "model": model, "auc": auc, "f1": f1,
        "y_pred": y_pred, "y_proba": y_proba
    }

# 5. COMPARE MODELS — ROC CURVES

plt.figure(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
print("\nSaved roc_curve_comparison.png")

# 6. FEATURE IMPORTANCE (Random Forest)

rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Random Forest Feature Importances:\n", importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

# 7. SUMMARY TABLE

summary = pd.DataFrame({
    name: {"ROC-AUC": res["auc"], "F1-Score": res["f1"]}
    for name, res in results.items()
}).T
print("\n=== Model Comparison Summary ===")
print(summary.sort_values("ROC-AUC", ascending=False))

best_model_name = summary["ROC-AUC"].idxmax()
print(f"\nBest model by ROC-AUC: {best_model_name}")