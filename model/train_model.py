import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import sys

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, 'data', 'dataset.xlsx')
MODEL_DIR  = os.path.join(BASE, 'model')
STATS_PATH = os.path.join(MODEL_DIR, 'model_stats.json')

print("=" * 60)
print("  FraudGuard AI — Hybrid Model Training")
print("=" * 60)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_excel(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,}  |  Normal: {(df['Class']==0).sum():,}")

X = df.drop('Class', axis=1)
y = df['Class']

# ── 2. Train/Test Split ────────────────────────────────────────────────────────
print("[2/7] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Scale ───────────────────────────────────────────────────────────────────
print("[3/7] Scaling features...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Isolation Forest (Anomaly Detection Layer) ─────────────────────────────
print("[4/7] Training Isolation Forest (anomaly layer)...")
iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
iso.fit(X_train_sc)
iso_train = iso.decision_function(X_train_sc).reshape(-1, 1)
iso_test  = iso.decision_function(X_test_sc).reshape(-1, 1)

# Add anomaly score as extra feature
X_train_hybrid = np.hstack([X_train_sc, iso_train])
X_test_hybrid  = np.hstack([X_test_sc,  iso_test])

# ── 5. Gradient Boosting Classifier (Main Model) ──────────────────────────────
print("[5/7] Training Gradient Boosting Classifier (main layer)...")
fraud_count  = y_train.sum()
normal_count = (y_train == 0).sum()
scale_weight = normal_count / fraud_count

model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train_hybrid, y_train)
print(f"      Training complete!")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
print("[6/7] Evaluating model...")
y_pred = model.predict(X_test_hybrid)
y_prob = model.predict_proba(X_test_hybrid)[:, 1]

roc_auc  = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
f1       = f1_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, output_dict=True)

print(f"\n  ROC-AUC Score : {roc_auc:.4f}")
print(f"  Avg Precision : {avg_prec:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  Precision     : {report['1']['precision']:.4f}")
print(f"  Recall        : {report['1']['recall']:.4f}")

# ROC Curve data
fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
# Precision-Recall data
prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_test, y_prob)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Feature importances
feat_names = list(X.columns) + ['anomaly_score']
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
top_features = [(feat_names[i], float(importances[i])) for i in top_idx]

# Transaction history sample (for dashboard)
df['fraud_prob'] = np.nan
test_idx = X_test.index
df.loc[test_idx, 'fraud_prob'] = y_prob

# ── 7. Save ────────────────────────────────────────────────────────────────────
print("[7/7] Saving models and stats...")
joblib.dump(model,  os.path.join(MODEL_DIR, 'gb_model.pkl'))
joblib.dump(iso,    os.path.join(MODEL_DIR, 'iso_forest.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

stats = {
    "roc_auc":       round(roc_auc, 4),
    "avg_precision": round(avg_prec, 4),
    "f1_score":      round(f1, 4),
    "precision":     round(report['1']['precision'], 4),
    "recall":        round(report['1']['recall'], 4),
    "total_transactions": int(len(df)),
    "total_fraud":    int(df['Class'].sum()),
    "total_normal":   int((df['Class'] == 0).sum()),
    "fraud_pct":      round(df['Class'].mean() * 100, 2),
    "true_positives": int(cm[1][1]),
    "false_positives": int(cm[0][1]),
    "false_negatives": int(cm[1][0]),
    "true_negatives": int(cm[0][0]),
    "roc_curve": {
        "fpr": [round(float(v), 4) for v in fpr[::5]],
        "tpr": [round(float(v), 4) for v in tpr[::5]]
    },
    "pr_curve": {
        "precision": [round(float(v), 4) for v in prec_arr[::5]],
        "recall":    [round(float(v), 4) for v in rec_arr[::5]]
    },
    "feature_importance": top_features,
    "confusion_matrix": cm.tolist(),
    "amount_stats": {
        "fraud_mean":  round(float(df[df['Class']==1]['Amount'].mean()), 4),
        "normal_mean": round(float(df[df['Class']==0]['Amount'].mean()), 4),
        "fraud_max":   round(float(df[df['Class']==1]['Amount'].max()), 4),
        "normal_max":  round(float(df[df['Class']==0]['Amount'].max()), 4),
    }
}

with open(STATS_PATH, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*60}")
print("  ✅ Training Complete! All files saved.")
print(f"  ROC-AUC: {roc_auc:.4f}  |  F1: {f1:.4f}  |  Recall: {report['1']['recall']:.4f}")
print(f"{'='*60}\n")
