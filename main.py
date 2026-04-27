"""
Explainable Anomaly Detection in IoT System Logs
=================================================
Using XGBoost, Random Forest, SMOTE, and SHAP on the BGL dataset.

Author : Faisal Ahamed Syed
        Auburn University at Montgomery
        fsyed8@aum.edu
"""

# ──────────────────────────────────────────────────────────────────────────
# SETUP — Install packages and mount Drive
# ──────────────────────────────────────────────────────────────────────────
!pip install drain3 shap imbalanced-learn xgboost -q

from google.colab import drive
drive.mount('/content/drive')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import shap

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
LOG_FILE     = "/content/drive/MyDrive/iot-anomaly/BGL.log"
OUTPUT_DIR   = "/content/drive/MyDrive/iot-anomaly"
SAMPLE_SIZE  = None      # None = full dataset; or e.g. 50000 for testing
SEED         = 42
TEST_SIZE    = 0.2
SHAP_SAMPLES = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD AND LABEL
# ──────────────────────────────────────────────────────────────────────────
print("[Stage 1] Loading BGL.log...")
rows = []
with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if SAMPLE_SIZE and i >= SAMPLE_SIZE:
            break
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        rows.append({
            "message": " ".join(parts[9:]),
            "label":   0 if parts[0] == "-" else 1,
        })
        if (i + 1) % 500_000 == 0:
            print(f"  Loaded {i+1:,} lines...")

df = pd.DataFrame(rows)
print(f"  Total: {len(df):,}  Anomaly rate: {df['label'].mean():.2%}")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 2 — DRAIN3 PARSING
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 2] Drain3 parsing... (20–30 min on full dataset)")
config = TemplateMinerConfig()
config.drain_sim_th = 0.4
config.drain_depth = 4
miner = TemplateMiner(config=config)

template_ids, template_strs = [], []
for i, msg in enumerate(df["message"]):
    res = miner.add_log_message(msg)
    template_ids.append(res["cluster_id"])
    template_strs.append(res["template_mined"])
    if (i + 1) % 500_000 == 0:
        print(f"  Parsed {i+1:,} / {len(df):,}  (templates so far: {len(set(template_ids))})")

df["template_id"]  = template_ids
df["template_str"] = template_strs
print(f"  Unique templates: {df['template_id'].nunique()}")

# Save checkpoint to Drive — protects against runtime resets
df.to_parquet(os.path.join(OUTPUT_DIR, "bgl_parsed.parquet"), index=False)
print("  Parsed dataset checkpoint saved.")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 3 — FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 3] Feature extraction...")
df["template_length"] = df["template_str"].apply(lambda x: len(x.split()))
df["has_error"] = df["message"].str.contains(
    r"error|fail|fatal|exception|crash", case=False, regex=True
).astype(int)
df["wildcard_count"] = df["template_str"].str.count(r"<\*>")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 4 — SPLIT + template_freq (training only)
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 4] Split and template_freq...")
X_base = df[["template_id", "template_length", "has_error", "wildcard_count"]]
y      = df["label"]

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_base, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

freq_map = X_train_b["template_id"].value_counts().to_dict()
X_train  = X_train_b.copy()
X_test   = X_test_b.copy()
X_train["template_freq"] = X_train["template_id"].map(freq_map).fillna(0).astype(int)
X_test ["template_freq"] = X_test ["template_id"].map(freq_map).fillna(0).astype(int)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 5 — SMOTE (training only)
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 5] SMOTE...")
smote = SMOTE(k_neighbors=5, random_state=SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: Normal {(y_train_sm==0).sum():,}  "
      f"Anomaly {(y_train_sm==1).sum():,}")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 6 — TRAIN 4 MODELS
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 6] Training 4 models...")

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    print(f"  {name}...")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, pred),  4),
        "Precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, pred,    zero_division=0), 4),
        "F1":        round(f1_score(y_te, pred,        zero_division=0), 4),
    }

results = [
    evaluate("RF  (no SMOTE)",
             RandomForestClassifier(n_estimators=20, random_state=SEED, n_jobs=-1),
             X_train, y_train, X_test, y_test),
    evaluate("RF  (SMOTE)",
             RandomForestClassifier(n_estimators=20, random_state=SEED, n_jobs=-1),
             X_train_sm, y_train_sm, X_test, y_test),
    evaluate("XGB (no SMOTE)",
             XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=100,
                           random_state=SEED, eval_metric="logloss", n_jobs=-1),
             X_train, y_train, X_test, y_test),
    evaluate("XGB (SMOTE)",
             XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=100,
                           random_state=SEED, eval_metric="logloss", n_jobs=-1),
             X_train_sm, y_train_sm, X_test, y_test),
]
results_df = pd.DataFrame(results)
print("\n=== RESULTS ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
print("\nResults saved to Drive.")

# ──────────────────────────────────────────────────────────────────────────
# STAGE 7 — SHAP ON BEST MODEL (XGB + SMOTE)
# ──────────────────────────────────────────────────────────────────────────
print("\n[Stage 7] SHAP analysis...")
features = ["template_id", "template_length", "has_error",
            "wildcard_count", "template_freq"]

best = XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=100,
                     random_state=SEED, eval_metric="logloss", n_jobs=-1)
best.fit(X_train_sm, y_train_sm)

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(X_test), size=SHAP_SAMPLES, replace=False)
X_sample = X_test.iloc[sample_idx]
y_sample = y_test.iloc[sample_idx]

explainer    = shap.TreeExplainer(best)
shap_values  = explainer.shap_values(X_sample)

# Plot 1 — global bar
shap.summary_plot(shap_values, X_sample, feature_names=features,
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), dpi=150)
plt.show()

# Plot 2 — beeswarm
shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=150)
plt.show()

# Plot 3 — waterfall (one anomaly)
anomaly_idx = (y_sample.values == 1).nonzero()[0][0]
shap.plots.waterfall(
    shap.Explanation(
        values        = shap_values[anomaly_idx],
        base_values   = explainer.expected_value,
        data          = X_sample.iloc[anomaly_idx].values,
        feature_names = features
    ),
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall.png"), dpi=150)
plt.show()

print("\nAll SHAP figures saved to Drive.")
print("Pipeline complete.")
