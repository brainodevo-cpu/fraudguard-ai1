from flask import Flask, jsonify, render_template, request, Response
import pandas as pd
import numpy as np
import joblib
import json
import os
import random
import time
import io

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE, 'model')
DATA_PATH  = os.path.join(BASE, 'data', 'dataset.xlsx')
STATS_PATH = os.path.join(MODEL_DIR, 'model_stats.json')

app = Flask(__name__)

def train_and_save_models():
    print("=" * 50)
    print("MODEL FILES NAHI MILE — AUTO TRAINING SHURU...")
    print("=" * 50)

    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, confusion_matrix,
        precision_score, recall_score, f1_score,
        average_precision_score, precision_recall_curve, roc_curve)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Dataset load ho raha hai...")
    df = pd.read_excel(DATA_PATH)

    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Scaler fit ho raha hai...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    print("Isolation Forest train ho raha hai...")
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X_sc)
    iso_scores = iso.decision_function(X_sc).reshape(-1, 1)

    X_hybrid = np.hstack([X_sc, iso_scores])

    print("Gradient Boosting train ho raha hai (thoda time lagega)...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gb.fit(X_hybrid, y)

    probs = gb.predict_proba(X_hybrid)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(y, probs)

    print(f"ROC-AUC: {auc:.4f}")

    cm               = confusion_matrix(y, preds).tolist()
    fpr_l, tpr_l, _  = roc_curve(y, probs)
    pr_p, pr_r, _    = precision_recall_curve(y, probs)
    fi = sorted(zip(X.columns, gb.feature_importances_),
                key=lambda x: x[1], reverse=True)

    def downsample(lst, n=200):
        lst = list(lst)
        if len(lst) <= n:
            return [round(float(v), 6) for v in lst]
        step = max(1, len(lst) // n)
        return [round(float(lst[i]), 6) for i in range(0, len(lst), step)]

    stats = {
        "roc_auc":            round(auc, 4),
        "total_transactions": int(len(df)),
        "total_fraud":        int(y.sum()),
        "total_normal":       int((y == 0).sum()),
        "fraud_pct":          round(float(y.mean()) * 100, 2),
        "precision":          round(precision_score(y, preds), 4),
        "recall":             round(recall_score(y, preds), 4),
        "f1_score":           round(f1_score(y, preds), 4),
        "avg_precision":      round(average_precision_score(y, probs), 4),
        "true_positives":     int(cm[1][1]),
        "confusion_matrix":   cm,
        "roc_curve":          {"fpr": downsample(fpr_l), "tpr": downsample(tpr_l)},
        "pr_curve":           {"precision": downsample(pr_p), "recall": downsample(pr_r)},
        "feature_importance": [[k, round(float(v), 6)] for k, v in fi],
        "amount_stats": {
            "fraud_mean":  round(float(df[y == 1]['Amount'].mean()), 4),
            "normal_mean": round(float(df[y == 0]['Amount'].mean()), 4),
        }
    }

    print("Models save ho rahe hain...")
    joblib.dump(gb,     os.path.join(MODEL_DIR, 'gb_model.pkl'))
    joblib.dump(iso,    os.path.join(MODEL_DIR, 'iso_forest.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print("AUTO TRAINING COMPLETE!")
    print("=" * 50)


gb_path  = os.path.join(MODEL_DIR, 'gb_model.pkl')
iso_path = os.path.join(MODEL_DIR, 'iso_forest.pkl')
sc_path  = os.path.join(MODEL_DIR, 'scaler.pkl')

if not (os.path.exists(gb_path) and os.path.exists(iso_path) and os.path.exists(sc_path)):
    train_and_save_models()
else:
    print("Model files already exist — checking stats...")
    try:
        with open(STATS_PATH) as f:
            _test = json.load(f)
        if 'confusion_matrix' not in _test:
            print("Purani stats — retraining for full stats...")
            train_and_save_models()
        else:
            print("Stats OK — training skip.")
    except:
        train_and_save_models()


print("Loading models...")
model  = joblib.load(gb_path)
iso    = joblib.load(iso_path)
scaler = joblib.load(sc_path)

with open(STATS_PATH) as f:
    STATS = json.load(f)

df_full = pd.read_excel(DATA_PATH)
X_full  = df_full.drop('Class', axis=1)
X_sc    = scaler.transform(X_full)
iso_sc  = iso.decision_function(X_sc).reshape(-1, 1)
X_hyb   = np.hstack([X_sc, iso_sc])
probs   = model.predict_proba(X_hyb)[:, 1]
df_full['fraud_prob'] = probs
df_full['predicted']  = (probs >= 0.5).astype(int)

_feat_cols   = list(X_full.columns)
_fraud_rows  = df_full[df_full['Class'] == 1][_feat_cols].values
_normal_rows = df_full[df_full['Class'] == 0][_feat_cols].values

print(f"Ready — {len(df_full):,} transactions | fraud pool: {len(_fraud_rows)} | normal pool: {len(_normal_rows)}")


def risk_label(p):
    return "HIGH" if p > 0.7 else "MEDIUM" if p > 0.4 else "LOW"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    return jsonify(STATS)

@app.route('/api/transactions')
def transactions():
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    ftype    = request.args.get('type', 'all')
    dff = df_full.copy()
    if ftype == 'fraud':    dff = dff[dff['Class'] == 1]
    elif ftype == 'normal': dff = dff[dff['Class'] == 0]
    total  = len(dff)
    subset = dff.iloc[(page-1)*per_page : page*per_page]
    records = []
    for i, row in subset.iterrows():
        records.append({
            "id":         f"TXN{str(i).zfill(6)}",
            "amount":     round(float(row['Amount']), 4),
            "time":       round(float(row['Time']), 2),
            "actual":     int(row['Class']),
            "predicted":  int(row['predicted']),
            "fraud_prob": round(float(row['fraud_prob']) * 100, 2),
            "risk":       risk_label(row['fraud_prob'])
        })
    return jsonify({"records": records, "total": total, "page": page,
                    "per_page": per_page, "total_pages": (total+per_page-1)//per_page})

@app.route('/api/predict', methods=['POST'])
def predict():
    data    = request.json
    row     = [float(data.get(c, 0)) for c in _feat_cols]
    arr     = np.array(row).reshape(1, -1)
    arr_sc  = scaler.transform(arr)
    iso_val = iso.decision_function(arr_sc).reshape(-1, 1)
    prob    = float(model.predict_proba(np.hstack([arr_sc, iso_val]))[0][1])
    return jsonify({"is_fraud": prob >= 0.5, "fraud_prob": round(prob*100,2), "risk": risk_label(prob)})

@app.route('/api/simulate', methods=['GET'])
def simulate():
    if random.random() < 0.30:
        row = _fraud_rows[random.randint(0, len(_fraud_rows)-1)]
    else:
        row = _normal_rows[random.randint(0, len(_normal_rows)-1)]
    arr     = row.reshape(1, -1)
    arr_sc  = scaler.transform(arr)
    iso_val = iso.decision_function(arr_sc).reshape(-1, 1)
    prob    = float(model.predict_proba(np.hstack([arr_sc, iso_val]))[0][1])
    amt_idx = _feat_cols.index('Amount')
    return jsonify({
        "txn_id":        f"SIM-{random.randint(100000,999999)}",
        "timestamp":     int(time.time() * 1000),
        "Amount":        round(float(row[amt_idx]), 4),
        "is_fraud":      prob >= 0.5,
        "fraud_prob":    round(prob*100, 2),
        "risk":          risk_label(prob),
        "anomaly_score": round(float(iso_val[0][0]), 4),
    })

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file. Send CSV as form-data key 'file'."}), 400
    try:
        df_in = pd.read_csv(io.BytesIO(request.files['file'].read()))
    except Exception as e:
        return jsonify({"error": f"Cannot parse CSV: {e}"}), 400
    required = [c for c in _feat_cols if c != 'Time']
    missing  = [c for c in required if c not in df_in.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400
    if 'Time' not in df_in.columns:
        df_in['Time'] = 0.0
    X_in     = df_in[_feat_cols].astype(float)
    X_in_sc  = scaler.transform(X_in)
    iso_in   = iso.decision_function(X_in_sc).reshape(-1, 1)
    probs_in = model.predict_proba(np.hstack([X_in_sc, iso_in]))[:, 1]
    preds_in = (probs_in >= 0.5).astype(int)
    records = [{"row": idx+1, "Amount": round(float(X_in.iloc[idx]['Amount']),4),
                "is_fraud": bool(preds_in[idx]), "fraud_prob": round(float(probs_in[idx])*100,2),
                "risk": risk_label(float(probs_in[idx])), "anomaly_score": round(float(iso_in[idx][0]),4)}
               for idx in range(len(df_in))]
    fraud_cnt = int(preds_in.sum())
    total     = len(records)
    return jsonify({"summary": {"total_rows": total, "fraud_count": fraud_cnt,
        "normal_count": total-fraud_cnt, "fraud_pct": round(fraud_cnt/total*100,2) if total else 0,
        "avg_fraud_prob": round(float(probs_in.mean())*100,2),
        "high_risk_count": sum(1 for r in records if r['risk']=='HIGH')}, "predictions": records})

@app.route('/api/batch_predict/download', methods=['POST'])
def batch_predict_download():
    if 'file' not in request.files:
        return jsonify({"error": "No file."}), 400
    try:
        df_in = pd.read_csv(io.BytesIO(request.files['file'].read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if 'Time' not in df_in.columns:
        df_in['Time'] = 0.0
    X_in     = df_in[_feat_cols].astype(float)
    X_in_sc  = scaler.transform(X_in)
    iso_in   = iso.decision_function(X_in_sc).reshape(-1, 1)
    probs_in = model.predict_proba(np.hstack([X_in_sc, iso_in]))[:, 1]
    preds_in = (probs_in >= 0.5).astype(int)
    df_out = df_in.copy()
    df_out['fraud_probability_pct'] = (probs_in*100).round(2)
    df_out['predicted_label']       = np.where(preds_in==1,'FRAUD','NORMAL')
    df_out['risk_level']            = [risk_label(p) for p in probs_in]
    df_out['anomaly_score']         = iso_in.flatten().round(4)
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=fraudguard_predictions.csv"})

@app.route('/api/amount_distribution')
def amount_distribution():
    return jsonify({
        "fraud":  [round(float(a),4) for a in df_full[df_full['Class']==1]['Amount'].tolist()],
        "normal": [round(float(a),4) for a in df_full[df_full['Class']==0]['Amount'].sample(500,random_state=42).tolist()]
    })

@app.route('/api/time_series')
def time_series():
    df_full['time_bin'] = pd.cut(df_full['Time'], bins=40, labels=False)
    grp = df_full.groupby('time_bin').agg(total=('Class','count'),fraud=('Class','sum')).reset_index()
    return jsonify({"bins": grp['time_bin'].tolist(), "total": grp['total'].tolist(), "fraud": grp['fraud'].tolist()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)