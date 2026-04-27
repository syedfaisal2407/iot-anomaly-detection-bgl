# Explainable Anomaly Detection in IoT System Logs

**Using XGBoost, Random Forest, SMOTE, and SHAP on the BGL Supercomputer Log Dataset**

> Faisal Ahamed Syed
> Department of Computer Information Systems and Cybersecurity
> Auburn University at Montgomery
> fsyed8@aum.edu

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange.svg)](https://colab.research.google.com/)

---

## Overview

This repository contains the complete implementation, results, and
documentation for a peer-reviewed quality study extending the IoT log
anomaly detection pipeline of Nguyen and Nguyen (CNIOT 2023) by
introducing two targeted enhancements:

1. **SMOTE** (Synthetic Minority Oversampling Technique) to correct the
   severe class imbalance in real-world log data
2. **SHAP** (SHapley Additive exPlanations) to provide post-hoc
   interpretability of model predictions

The full pipeline is evaluated on the **Blue Gene/L (BGL) supercomputer log
dataset** comprising 4,747,963 entries with a 7.34% anomaly rate.

---

## Key Results

| Model            | Accuracy | Precision | Recall     | F1     |
|------------------|----------|-----------|------------|--------|
| RF (no SMOTE)    | 0.9995   | 1.0000    | 0.9932     | 0.9966 |
| RF (SMOTE)       | 0.9993   | 0.9913    | 0.9995     | 0.9954 |
| XGB (no SMOTE)   | 0.9995   | 1.0000    | 0.9933     | 0.9966 |
| **XGB (SMOTE)**  | 0.9993   | 0.9912    | **0.9995** | 0.9954 |

- **SMOTE improved Recall from 0.9932 to 0.9995** for both classifiers
- **434 additional real anomalies caught** at the cost of 613 false alarms
- **Outperforms LogAnomaly BGL benchmark** (96% F1) at 99.54% F1
- **SHAP identifies template_freq as #1 feature** — rare templates are the
  dominant anomaly signal

---

## Repository Structure

```
iot-anomaly-detection-bgl/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── main.ipynb                  # Complete Colab notebook (all 7 stages)
├── results/
│   ├── results.csv                 # 4-model comparison results
│   ├── shap_importance.png         # Global feature importance
│   ├── shap_beeswarm.png           # Beeswarm directional plot
│   └── shap_waterfall.png          # Single-instance explanation
├── paper/
│   └── Final_Report.tex            # IEEE conference paper LaTeX source
└── docs/
    └── methodology.md              # Detailed methodology notes
```

---

## Quick Start (Google Colab — Recommended)

The fastest way to reproduce the results is via Google Colab.

### Prerequisites
- A Google account (for Colab and Drive)
- ~2 GB free space on Google Drive (for BGL.log + Parquet checkpoint)
- Stable internet connection (parsing takes 30–40 minutes)

### Step 1 — Download the BGL dataset

Download `BGL.zip` (~709 MB) from Zenodo:
**https://zenodo.org/records/8196385/files/BGL.zip?download=1**

### Step 2 — Upload to Google Drive

1. Go to https://drive.google.com
2. Create a folder called `iot-anomaly`
3. Upload `BGL.zip` into that folder

### Step 3 — Open the notebook in Colab

1. Click the Colab badge at the top of this README, or open
   https://colab.research.google.com directly
2. File → Upload notebook → select `notebooks/main.ipynb`
3. Run all cells in order

The notebook automatically:
- Mounts your Google Drive
- Installs all dependencies
- Unzips BGL.zip
- Runs all 7 pipeline stages
- Saves results and SHAP figures back to Drive

---

## Local Installation (Alternative)

If you prefer to run locally instead of Colab, you need at least 16 GB RAM.

```bash
# Clone the repository
git clone https://github.com/syedfaisal2407/iot-anomaly-detection-bgl.git
cd iot-anomaly-detection-bgl

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download the BGL dataset manually from Zenodo and extract BGL.log
# https://zenodo.org/records/8196385/files/BGL.zip?download=1

# Open Jupyter
jupyter notebook notebooks/main.ipynb
```

Before running, edit Stage 1 in the notebook to point at your local
`BGL.log` path:

```python
LOG_FILE = "/path/to/your/BGL.log"
```

---

## Pipeline Stages

The notebook executes seven sequential stages:

| Stage | What It Does                                       | Output                            |
|-------|----------------------------------------------------|-----------------------------------|
| 1     | Load BGL.log and parse labels                       | DataFrame of message + label      |
| 2     | Drain3 parsing → 1,823 unique templates             | template_id, template_str         |
| 3     | Extract 5 features                                  | feature columns added             |
| 4     | 80/20 stratified split + template_freq mapping     | X_train, X_test, y_train, y_test  |
| 5     | Apply SMOTE to training partition only             | X_train_sm, y_train_sm            |
| 6     | Train 4 models, evaluate on test set               | results.csv                       |
| 7     | SHAP TreeExplainer on best model                    | 3 SHAP figures                    |

Total runtime in Colab (free tier): approximately **60–90 minutes**.

---

## Methodology Highlights

### Dataset
- **Source:** Loghub collection (Zenodo doi:10.5281/zenodo.8196385)
- **Size:** 4,747,963 log entries, 214.7 days from Lawrence Livermore
  National Laboratory
- **Class distribution:** 92.66% normal, 7.34% anomalous

### Features (5)
1. `template_id` — Drain3 cluster identifier (categorical)
2. `template_length` — token count in template (discrete)
3. `has_error` — keyword indicator (binary)
4. `wildcard_count` — number of `<*>` placeholders (discrete)
5. `template_freq` — occurrence count in training (continuous, prevents
   leakage by computing on training only)

### SMOTE
- k-neighbours = 5
- Applied to training partition only (after stratified split)
- Random seed = 42 throughout

### Classifiers
- **Random Forest:** n_estimators=20, seed=42
- **XGBoost:** lr=0.3, max_depth=6, n_estimators=100, seed=42

Hyperparameters match Nguyen & Nguyen (2023) for direct comparability.

### SHAP
- Method: TreeExplainer (exact Shapley values, polynomial time)
- Sample size: 5,000 stratified test entries
- Outputs: global bar chart, beeswarm, waterfall (one anomalous instance)

---

## Reproducibility

All randomness is controlled by **random seed 42** throughout:
- Train/test split (`sklearn`)
- SMOTE oversampling (`imbalanced-learn`)
- Random Forest (`scikit-learn`)
- XGBoost
- SHAP test sample selection (`numpy`)

Running the notebook end-to-end produces the exact results in
`results/results.csv` and the three figures in `results/`.

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@misc{syed2025bgl,
  author       = {Faisal Ahamed Syed},
  title        = {Explainable Anomaly Detection in IoT System Logs Using
                  XGBoost, Random Forest, SMOTE, and SHAP},
  year         = {2025},
  institution  = {Auburn University at Montgomery},
  howpublished = {\url{https://github.com/syedfaisal2407/iot-anomaly-detection-bgl}}
}
```

---

## Related Work

This study extends and builds upon:

1. **Nguyen & Nguyen (2023)** — Baseline IoT log pipeline, ACM CNIOT
2. **Alam et al. SXAD (2024)** — Closest competitor, applies SHAP to log
   data, IEEE Access
3. **Talukder et al. (2024)** — SMOTE on XGBoost/RF for security data,
   J. Big Data
4. **Meng et al. LogAnomaly (2019)** — Published BGL benchmark (96% F1),
   IJCAI
5. **Du et al. DeepLog (2017)** — LSTM-based log anomaly detection,
   ACM CCS
6. **Chawla et al. (2002)** — Original SMOTE paper, JAIR
7. **Lundberg & Lee (2017)** — Original SHAP paper, NeurIPS
8. **He et al. Drain (2017)** — Log parsing algorithm, IEEE ICWS

See `paper/Final_Report.tex` for the full literature review.

---

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE)
for details.

The BGL dataset is licensed separately by its original distributors.
Refer to the Loghub collection for dataset-specific terms.

---

## Acknowledgements

The author thanks the Department of Computer Information Systems and
Cybersecurity at Auburn University at Montgomery for supporting this
research.

---

## Contact

For questions about this work:

**Faisal Ahamed Syed**
fsyed8@aum.edu
Auburn University at Montgomery
