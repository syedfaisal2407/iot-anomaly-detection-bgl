#!/usr/bin/env python3
"""
Explainable Anomaly Detection in IoT System Logs
=================================================
XGBoost, Random Forest, SMOTE, and SHAP on the BGL supercomputer log dataset.

Author : Faisal Ahamed Syed
         Department of Computer Information Systems and Cybersecurity
         Auburn University at Montgomery
         fsyed8@aum.edu

Usage:
    python main.py --log-file /path/to/BGL.log --output-dir results/
    python main.py --log-file BGL.log --sample-size 50000   # quick test run

For the Google Colab version of this pipeline, see notebooks/main.ipynb.
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURES = ["template_id", "template_length", "has_error",
            "wildcard_count", "template_freq"]


# ──────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD AND LABEL
# ──────────────────────────────────────────────────────────────────────────
def load_and_label(log_file: str, sample_size: int | None) -> pd.DataFrame:
    """Parse raw BGL.log lines into (message, label) rows.

    BGL convention: a leading "-" marks a normal line; anything else is the
    alert category, i.e. an anomaly.
    """
    print("[Stage 1] Loading BGL.log...")
    rows = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
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
                print(f"  Loaded {i + 1:,} lines...")

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df):,}  Anomaly rate: {df['label'].mean():.2%}")
    return df


# ──────────────────────────────────────────────────────────────────────────
# STAGE 2 — DRAIN3 PARSING
# ──────────────────────────────────────────────────────────────────────────
def parse_templates(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Mine log templates with Drain3 and checkpoint the parsed frame."""
    print("\n[Stage 2] Drain3 parsing... (20-30 min on full dataset)")
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
            print(f"  Parsed {i + 1:,} / {len(df):,}  "
                  f"(templates so far: {len(set(template_ids))})")

    df["template_id"] = template_ids
    df["template_str"] = template_strs
    print(f"  Unique templates: {df['template_id'].nunique()}")

    checkpoint = os.path.join(output_dir, "bgl_parsed.parquet")
    df.to_parquet(checkpoint, index=False)
    print(f"  Parsed dataset checkpoint saved to {checkpoint}")
    return df


# ──────────────────────────────────────────────────────────────────────────
# STAGE 3 — FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 3] Feature extraction...")
    df["template_length"] = df["template_str"].apply(lambda x: len(x.split()))
    df["has_error"] = df["message"].str.contains(
        r"error|fail|fatal|exception|crash", case=False, regex=True
    ).astype(int)
    df["wildcard_count"] = df["template_str"].str.count(r"<\*>")
    return df


# ──────────────────────────────────────────────────────────────────────────
# STAGE 4 — SPLIT + template_freq (training only, prevents leakage)
# ──────────────────────────────────────────────────────────────────────────
def split_and_freq(df: pd.DataFrame, test_size: float, seed: int):
    print("\n[Stage 4] Split and template_freq...")
    X_base = df[["template_id", "template_length", "has_error",
                 "wildcard_count"]]
    y = df["label"]

    X_train_b, X_test_b, y_train, y_test = train_test_split(
        X_base, y, test_size=test_size, stratify=y, random_state=seed
    )

    freq_map = X_train_b["template_id"].value_counts().to_dict()
    X_train = X_train_b.copy()
    X_test = X_test_b.copy()
    X_train["template_freq"] = (X_train["template_id"].map(freq_map)
                                .fillna(0).astype(int))
    X_test["template_freq"] = (X_test["template_id"].map(freq_map)
                               .fillna(0).astype(int))
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────
# STAGE 5 — SMOTE (training partition only)
# ──────────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train, seed: int):
    print("\n[Stage 5] SMOTE...")
    smote = SMOTE(k_neighbors=5, random_state=seed)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: Normal {(y_train_sm == 0).sum():,}  "
          f"Anomaly {(y_train_sm == 1).sum():,}")
    return X_train_sm, y_train_sm


# ──────────────────────────────────────────────────────────────────────────
# STAGE 6 — TRAIN 4 MODELS
# ──────────────────────────────────────────────────────────────────────────
def _evaluate(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    print(f"  {name}...")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, pred), 4),
        "Precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, pred, zero_division=0), 4),
        "F1":        round(f1_score(y_te, pred, zero_division=0), 4),
    }


def _rf(seed):
    return RandomForestClassifier(n_estimators=20, random_state=seed,
                                  n_jobs=-1)


def _xgb(seed):
    return XGBClassifier(learning_rate=0.3, max_depth=6, n_estimators=100,
                         random_state=seed, eval_metric="logloss", n_jobs=-1)


def train_models(X_train, X_test, y_train, y_test,
                 X_train_sm, y_train_sm, output_dir: str, seed: int):
    print("\n[Stage 6] Training 4 models...")
    results = [
        _evaluate("RF  (no SMOTE)", _rf(seed),
                  X_train, y_train, X_test, y_test),
        _evaluate("RF  (SMOTE)", _rf(seed),
                  X_train_sm, y_train_sm, X_test, y_test),
        _evaluate("XGB (no SMOTE)", _xgb(seed),
                  X_train, y_train, X_test, y_test),
        _evaluate("XGB (SMOTE)", _xgb(seed),
                  X_train_sm, y_train_sm, X_test, y_test),
    ]
    results_df = pd.DataFrame(results)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    out = os.path.join(output_dir, "results.csv")
    results_df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


# ──────────────────────────────────────────────────────────────────────────
# STAGE 7 — SHAP ON BEST MODEL (XGB + SMOTE)
# ──────────────────────────────────────────────────────────────────────────
def shap_analysis(X_train_sm, y_train_sm, X_test, y_test,
                  output_dir: str, seed: int, shap_samples: int):
    print("\n[Stage 7] SHAP analysis...")
    best = _xgb(seed)
    best.fit(X_train_sm, y_train_sm)

    n = min(shap_samples, len(X_test))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_sample = y_test.iloc[sample_idx]

    explainer = shap.TreeExplainer(best)
    shap_values = explainer.shap_values(X_sample)

    # Plot 1 — global bar
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance.png"), dpi=150)
    plt.close()

    # Plot 2 — beeswarm
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES,
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=150)
    plt.close()

    # Plot 3 — waterfall (first anomalous instance in sample)
    anomalies = (y_sample.values == 1).nonzero()[0]
    if len(anomalies) == 0:
        print("  No anomalies in SHAP sample; skipping waterfall plot.")
        return
    anomaly_idx = anomalies[0]
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[anomaly_idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[anomaly_idx].values,
            feature_names=FEATURES,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall.png"), dpi=150)
    plt.close()
    print(f"  All SHAP figures saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explainable anomaly detection on BGL logs "
                    "(Drain3 + SMOTE + RF/XGBoost + SHAP)."
    )
    p.add_argument("--log-file", required=True,
                   help="Path to BGL.log (download from Zenodo, see README)")
    p.add_argument("--output-dir", default="output",
                   help="Directory for results, checkpoints, and figures "
                        "(default: output/)")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Limit to first N lines for a quick test run "
                        "(default: full dataset)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test split fraction (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used everywhere (default: 42)")
    p.add_argument("--shap-samples", type=int, default=5000,
                   help="Number of test rows for SHAP analysis "
                        "(default: 5000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_label(args.log_file, args.sample_size)
    df = parse_templates(df, args.output_dir)
    df = extract_features(df)
    X_train, X_test, y_train, y_test = split_and_freq(
        df, args.test_size, args.seed)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, args.seed)
    train_models(X_train, X_test, y_train, y_test,
                 X_train_sm, y_train_sm, args.output_dir, args.seed)
    shap_analysis(X_train_sm, y_train_sm, X_test, y_test,
                  args.output_dir, args.seed, args.shap_samples)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
