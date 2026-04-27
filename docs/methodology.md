# Methodology — Detailed Notes

This document expands on the methodology summarised in the README and
formally specified in the conference paper (`paper/Final_Report.tex`).

---

## 1. BGL Log Format

Each raw BGL log line follows this structure:

```
LABEL  TIMESTAMP  DATE  NODE  DATETIME  TYPE  COMPONENT  LEVEL  MESSAGE
```

Example (normal):
```
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779
  R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

Example (anomalous):
```
APPREAD 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.51
  R02-M1-N0-C:J12-U11 APP FATAL ciod: failed to read message prefix
```

Labelling rule: **dash (`-`) = normal, anything else = anomaly.**

---

## 2. Drain3 Configuration

| Parameter         | Value | Rationale                              |
|-------------------|-------|----------------------------------------|
| `drain_sim_th`    | 0.4   | Matches Nguyen & Nguyen (2023)         |
| `drain_depth`     | 4     | Matches Nguyen & Nguyen (2023)         |
| `drain_max_children` | 100 (default) | Sufficient for BGL diversity      |

Result: **1,823 unique templates** discovered from 4,747,963 messages.

---

## 3. Feature Engineering Decisions

### Why these 5 features?

- **template_id** captures the structural category (what type of event)
- **template_length** approximates message complexity
- **has_error** is the most intuitive lexical signal (turned out weakest)
- **wildcard_count** captures variability in the template itself
- **template_freq** captures rarity — turned out to be the strongest signal

### Why template_freq is computed on training only?

If frequency were computed on the full dataset, the model would have access
to information from the test set during training — a form of data leakage.
Computing on training only ensures the model evaluates as it would in
deployment, where future log frequencies are unknown.

---

## 4. Train/Test Split

- **Strategy:** Stratified, 80/20
- **Random seed:** 42
- **Result:** 3,798,370 train / 949,593 test
- **Class ratio preserved** in both partitions at 7.34%

---

## 5. SMOTE Application

### Critical rule
SMOTE is applied **after** the split, **only** to the training partition.
Applying before the split would generate synthetic test samples = data
leakage = inflated Recall numbers.

### Configuration
- `k_neighbors = 5`
- `random_state = 42`

### Result
- Before: 3,519,602 normal / 278,768 anomaly (ratio 7.34%)
- After:  3,519,602 normal / 3,519,602 anomaly (ratio 50%)

### Known limitation
SMOTE was designed for continuous features. Two of our features are
categorical/binary:
- `template_id` — interpolated values may not map to real templates
- `has_error` — values between 0 and 1 are meaningless

Despite this, empirical Recall improvement is observed for both classifiers.
This is documented in the paper's Limitations section as a threat to
construct validity.

---

## 6. Classifier Hyperparameters

Hyperparameters match Nguyen & Nguyen (2023) for direct comparability.

### Random Forest
- `n_estimators = 20`
- `random_state = 42`
- All other parameters at scikit-learn defaults

### XGBoost
- `learning_rate = 0.3`
- `max_depth = 6`
- `n_estimators = 100`
- `random_state = 42`
- `eval_metric = "logloss"`

---

## 7. SHAP Analysis

### Why TreeExplainer instead of Kernel SHAP?

| Property        | Kernel SHAP        | TreeExplainer        |
|-----------------|--------------------|----------------------|
| Model type      | Model-agnostic     | Tree-models only     |
| Speed           | Slow               | Fast                 |
| Accuracy        | Approximate        | Exact Shapley values |
| Best for        | Any model          | XGBoost, RF, etc.    |

We use TreeExplainer because we evaluate XGBoost and Random Forest, both of
which are tree-based ensembles. SXAD (Alam et al. 2024) used Kernel SHAP
even though their models were tree-based — this is a methodological
difference where this work is more rigorous.

### Sample size
SHAP is computed on a stratified 5,000-entry sample of the test set due to
runtime constraints in the Colab environment. Full-test SHAP would take
several hours.

---

## 8. Evaluation Metrics

### Primary: Recall

Recall = TP / (TP + FN)

In security contexts, missing a real anomaly (false negative) costs more
than investigating a false alarm (false positive). Recall directly measures
how many real threats are caught.

### Secondary: Precision, F1, Accuracy

All four are reported for completeness and comparability with prior
benchmarks (e.g., LogAnomaly's 96% F1 on BGL).

---

## 9. Reproducibility

To reproduce these exact results:

1. Use the BGL dataset from Zenodo (DOI: 10.5281/zenodo.8196385)
2. Use Python 3.10
3. Install packages from `requirements.txt`
4. Run `notebooks/main.ipynb` cell-by-cell in order
5. Confirm `random_state = 42` is fixed throughout

Expected output: `results.csv` matching the table in the README.
