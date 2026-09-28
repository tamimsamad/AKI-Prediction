import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


DATA_PATH = "data/AKI_Dataset/Perspective_scrb_3.csv"

FEATURES = [
    "X0", "X1", "X2", "X3", "X4", "X5", "X6",
    "X8", "X9", "X10", "X11", "X12", "X13",
    "X14", "X15", "X16", "X17", "X18", "X19",
    "X20", "X21",
]

TARGET = "X1917"


# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]


# -------------------------
# 2. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


# -------------------------
# 3. Logistic Regression
# -------------------------
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

logistic_model.fit(X_train, y_train)

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
logistic_pred = (logistic_prob >= 0.50).astype(int)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred, digits=3))

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, logistic_prob), 3),
)

print(
    "Average Precision:",
    round(average_precision_score(y_test, logistic_prob), 3),
)


# -------------------------
# 4. Random Forest
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= 0.50).astype(int)

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, digits=3))

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, rf_prob), 3),
)

print(
    "Average Precision:",
    round(average_precision_score(y_test, rf_prob), 3),
)


# -------------------------
# 5. Precision-Recall Curve
# -------------------------
logistic_precision, logistic_recall, _ = precision_recall_curve(
    y_test,
    logistic_prob,
)

rf_precision, rf_recall, _ = precision_recall_curve(
    y_test,
    rf_prob,
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_recall,
    logistic_precision,
    label="Logistic Regression",
)

plt.plot(
    rf_recall,
    rf_precision,
    label="Random Forest",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("models/precision_recall_curve.png", dpi=300)

plt.show()


# -------------------------
# 6. Random Forest Feature Importance
# -------------------------
importance = pd.Series(
    rf_model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string())