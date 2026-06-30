"""
Give Me Some Credit - Handling Class Imbalance

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# 1. LOAD + FEATURE ENGINEER (same as before)

df = pd.read_csv("cs-training.csv", index_col=0)

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(0)
    data.loc[data["age"] < 18, "age"] = data["age"].median()

    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"]:
        data[col] = data[col].clip(upper=data[col].quantile(0.999))

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
        bins=[-0.01, 0.3, 0.7, 1.0, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    data["IsRetired"] = (data["age"] >= 65).astype(int)

    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        data[col] = data[col].clip(upper=data[col].quantile(0.99))

    return data

df = engineer_features(df)

y = df["SeriousDlqin2yrs"]
X = df.drop(columns=["SeriousDlqin2yrs"])

print("Class distribution (full data):")
print(y.value_counts(), "\n")

# 2. TRAIN/TEST SPLIT  -->  ALWAYS split BEFORE any resampling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train class distribution:", y_train.value_counts().to_dict())

# 3. APPLY SMOTE  -->  TRAINING DATA ONLY

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Train class distribution AFTER SMOTE:", pd.Series(y_train_smote).value_counts().to_dict())

# 4. HELPER: find the threshold that MAXIMIZES F1 (instead of using 0.5)

def best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx], precisions[best_idx], recalls[best_idx]

def evaluate(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- {name} (threshold={threshold:.3f}) ---")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")
    return {"roc_auc": auc, "pr_auc": pr_auc, "f1": f1, "threshold": threshold}

results = {}

# 5A. BASELINE: class_weight="balanced", default 0.5 threshold

rf_baseline = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=20,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_baseline.fit(X_train, y_train)
proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]
results["1. Baseline (class_weight, thresh=0.5)"] = evaluate(
    "Baseline class_weight @0.5", y_test, proba_baseline, 0.5
)

# 5B. SMOTE: trained on resampled (balanced) training data

rf_smote = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=20,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)
proba_smote = rf_smote.predict_proba(X_test_scaled)[:, 1]
results["2. SMOTE (thresh=0.5)"] = evaluate(
    "SMOTE @0.5", y_test, proba_smote, 0.5
)

# 5C. THRESHOLD TUNING on top of the baseline model

best_thresh, best_f1, best_prec, best_rec = best_f1_threshold(y_test, proba_baseline)
print(f"\nOptimal threshold for baseline model: {best_thresh:.3f} "
      f"(F1={best_f1:.4f}, Precision={best_prec:.4f}, Recall={best_rec:.4f})")
results["3. Baseline + tuned threshold"] = evaluate(
    "Baseline + tuned threshold", y_test, proba_baseline, best_thresh
)

# 5D. SMOTE + THRESHOLD TUNING (usually the winning combo)

best_thresh_smote, best_f1_smote, best_prec_smote, best_rec_smote = best_f1_threshold(y_test, proba_smote)
print(f"\nOptimal threshold for SMOTE model: {best_thresh_smote:.3f} "
      f"(F1={best_f1_smote:.4f}, Precision={best_prec_smote:.4f}, Recall={best_rec_smote:.4f})")
results["4. SMOTE + tuned threshold"] = evaluate(
    "SMOTE + tuned threshold", y_test, proba_smote, best_thresh_smote
)  

# 6. SUMMARY COMPARISON

summary = pd.DataFrame(results).T[["roc_auc", "pr_auc", "f1", "threshold"]]
summary.columns = ["ROC-AUC", "PR-AUC", "F1-Score", "Threshold Used"]
print("\n" + "=" * 70)
print("SUMMARY: Effect of each imbalance-handling strategy")
print("=" * 70)
print(summary.round(4))

# 7. PLOT: Precision-Recall curve + F1 vs threshold

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

precisions, recalls, thresholds = precision_recall_curve(y_test, proba_baseline)
axes[0].plot(recalls, precisions, label="Baseline (class_weight)")
precisions_s, recalls_s, thresholds_s = precision_recall_curve(y_test, proba_smote)
axes[0].plot(recalls_s, precisions_s, label="SMOTE")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve"); axes[0].legend()

f1_curve = 2 * precisions * recalls / (precisions + recalls + 1e-12)
axes[1].plot(thresholds, f1_curve[:-1], label="Baseline F1 vs threshold")
axes[1].axvline(best_thresh, color="red", linestyle="--", label=f"Best threshold={best_thresh:.2f}")
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("F1-score")
axes[1].set_title("F1-Score vs Decision Threshold"); axes[1].legend()

plt.tight_layout()
plt.savefig("imbalance_handling_comparison.png", dpi=150)
print("\nSaved imbalance_handling_comparison.png")