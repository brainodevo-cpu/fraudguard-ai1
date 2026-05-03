# 🛡️ FraudGuard AI — Command Center

A full-stack fraud detection dashboard using a **Hybrid ML Model** (Gradient Boosting + Isolation Forest).

---

## 📁 Project Structure

```
FraudGuardAI/
├── data/
│   └── dataset.xlsx          ← Your CSV/Excel dataset
├── model/
│   ├── train_model.py        ← Run this FIRST to train the model
│   ├── gb_model.pkl          ← (auto-created after training)
│   ├── iso_forest.pkl        ← (auto-created after training)
│   ├── scaler.pkl            ← (auto-created after training)
│   └── model_stats.json      ← (auto-created after training)
├── templates/
│   └── index.html            ← Full frontend UI
├── app.py                    ← Flask backend server
├── requirements.txt          ← Python dependencies
└── README.md
```

---

## ⚡ HOW TO RUN — Step by Step

### STEP 1 — Install Python
Make sure Python 3.9+ is installed.
Download from: https://www.python.org/downloads/

### STEP 2 — Open Terminal / Command Prompt
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac/Linux**: Open Terminal

Navigate to this folder:
```bash
cd path/to/FraudGuardAI
```
Example (Windows):
```bash
cd C:\Users\YourName\Downloads\FraudGuardAI
```

### STEP 3 — Install Required Libraries
```bash
pip install -r requirements.txt
```
Wait for all packages to install. This takes 1-2 minutes.

### STEP 4 — Train the Model (ONLY ONCE)
```bash
python model/train_model.py
```
You will see output like:
```
✅ Training Complete!
ROC-AUC: 0.9816 | F1: 0.9000 | Recall: 0.8526
```
This creates the model files automatically.

### STEP 5 — Start the Web Server
```bash
python app.py
```
You will see:
```
* Running on http://127.0.0.1:5000
```

### STEP 6 — Open the Website
Open your browser and go to:
```
http://localhost:5000
```

🎉 **FraudGuard AI is now running!**

---

## 📊 Features

| Page | What it shows |
|------|--------------|
| **Overview** | KPI cards, class distribution, confusion matrix, time series |
| **Performance** | ROC Curve, Precision-Recall Curve, Feature Importance, Metrics Bar |
| **Transaction History** | All transactions with fraud probability, filter by fraud/normal |
| **Predict** | Enter transaction values and get real-time fraud prediction |
| **Audit Ledger** | All detected fraud transactions |
| **Model Info** | Architecture details, dataset info, training config |

---

## 🤖 Model Architecture

**Layer 1 — Isolation Forest** (Unsupervised)
- Detects statistically anomalous transactions
- Outputs anomaly score as extra feature
- n_estimators=200, contamination=3%

**Layer 2 — Gradient Boosting Classifier** (Supervised)
- Learns fraud patterns from labeled data + anomaly scores
- n_estimators=300, max_depth=5, learning_rate=0.05
- Train/test split: 80/20 with stratification

**Results:**
- ROC-AUC: 98.16%
- F1 Score: 90.00%
- Recall: 85.26%
- Precision: 95.29%

---

## ❓ Common Issues

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"Port already in use":**
```bash
python app.py
# If port 5000 busy, change in app.py: port=5001
```

**Model files not found:**
```bash
python model/train_model.py
```
(Run training before starting the server)
