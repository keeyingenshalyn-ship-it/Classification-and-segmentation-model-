# Grant Analysis in the Creative Industry

Does receiving a financial grant help creative-industry firms survive and attract follow-on
equity? This project uses **pre-grant** financial and operational data to predict two
**post-grant** outcomes — **firm survival** and **follow-on equity investment** — across three
funding cohorts:

| Cohort | Description | n (firm-level) |
|---|---|---|
| **A** | Non-tech companies | 169 |
| **B** | AI & Data Economy (software / data) | 150 |
| **C** | Clean Growth (clean tech) | 299 |

All analysis lives in [`finals-2.ipynb`](finals-2.ipynb).

---

## Research questions

1. **Survival** — can pre-grant financial health predict whether a firm survives 36 / 60 months after the grant?
2. **Follow-on equity** — can pre-grant features predict whether a firm raises follow-on equity within 3 years? (Cohort-B only; see data caveats.)
3. **Segmentation** — what natural risk segments exist among grant recipients, and does sector or financial health drive them?

---

## Data

The notebook expects an Excel workbook as its raw input and derives a cleaned firm-level CSV from it:

| File | Role |
|---|---|
| `Unified_Dataset_DG_cleaned.xlsx` (`Sheet1`) | **Raw input** — 1,017 grant-level rows across cohorts A/B/C |
| `firm_level_clean.csv` | **Produced** — 618 firms (one row per `CRN_final`), 61 columns; the modelling frame |

> ⚠️ The notebook currently uses a **hard-coded absolute path**:
> `/Users/shalynkee/Downloads/predcitive analytics /dissertiaion /Unified_Dataset_DG_cleaned.xlsx`.
> Update `PATH` in the first code cell to point at your own copy of the workbook before running.

### Data caveats (important for interpreting results)

- **Two outcome families with different coverage.** DG-derived survival columns cover **all 1,017 firms / all 3 cohorts**; Beauhurst-derived survival and **all equity columns are sparse (~100 firms) and concentrated in Cohort B**. Survival uses the DG columns; equity is treated as a **Cohort-B-only exploratory sub-study**.
- **Severe class imbalance.** 5-year survival is ~92% positive; 12m/36m survival ≈ 99.7% / 97.9% positive. Plain accuracy is meaningless — the analysis reports **PR-AUC and minority-class (failure) recall**.
- **Rare-event equity target.** `follow_on_equity_3yr` has only **9 positives among 91 firms**, supporting roughly **one predictor**. It is framed as association + risk-ranking, not a deployable classifier.
- **Missing ≠ 0.** Equity is unobserved (not zero) outside Cohort B, so NaN is never imputed as 0 across cohorts.

---

## Notebook structure

| Section | What it does |
|---|---|
| **1. EDA (raw data)** | Load workbook; profile dtypes, missingness, numeric distributions; map outcome coverage by cohort. |
| **2. Data cleaning** | Normalise messy values/types; collapse grant-rows → one row per firm (`CRN_final`); quarantine leakage; drop >80%-missing predictors → **`firm_level_clean.csv`**. |
| **3. EDA (firm level)** | Outcome prevalence, pre-grant feature distributions, categorical survival rates, correlations, bivariate signal. |
| **4. Clustering (k-means)** | Leakage-safe scaled matrix `X_cluster` (618×7); choose *k* via elbow + silhouette; fit **k=4**; profile segments. |
| **5. Mixed-type segmentation (k-prototypes)** | Add `cohort`, `enterprise_size`, `Legal form` as native categoricals; profile financial × business-type segments. |
| **6. Visualisation** | PCA 2-D map (PC1+PC2 ≈ 49.5% variance) + per-segment "fingerprint" heatmaps. |
| **7. Feature engineering** | Derived leakage-safe pre-grant features: financial ratios, award/grant intensity, per-capita efficiency, skew-tamed size terms, missingness flags. |
| **8. Classification — 36-month survival** | Logistic regression + random forest; imbalance handling; odds-ratio & **SHAP** interpretation; robustness (seed stability, bootstrap CIs, imbalance-method sensitivity, VIF). |
| **9. Hypothesis tests** | H1: pre-grant financial health ↔ 36m survival (LR test). H2: Cohort A survival < Cohort B. |
| **10. Follow-on equity (Cohort B)** | Penalised/Firth-style logistic, LOO validation, bootstrap CIs, permutation test; **elastic-net** headline model + ridge for full coefficient interpretation. |

---

## Key findings

- **Financial health, not sector, is the dominant axis.** k-means and k-prototypes recover essentially the same four segments; adding cohort/business-type does not reshuffle firms.
- **Solvency + maturity drive survival.** Mature, well-capitalised firms survive at ~99%; young, highly-leveraged, negative-equity firms at ~88% (highest dissolution rate). Company **age** and **net assets / leverage** separate winners from at-risk firms.
- **Weak but real survival signal.** Pre-grant financial features carry a statistically meaningful but weak signal for early failure (PR-AUC ≈ 15× baseline); with only ~13 failures the estimate is unstable and best used for **risk-ranking**, not hard classification.
- **Equity headline model:** elastic-net logistic — **ROC-AUC 0.712, PR-AUC 0.187** (vs 0.5 / 0.101 baseline) — interpretable and regularised for the small sample. Company age is the dominant driver.

---

## Getting started

### Requirements

Python 3.13 (developed on the Anaconda `base` env). Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels imbalanced-learn kmodes shap
```

| Package | Used for |
|---|---|
| pandas, numpy | data wrangling |
| matplotlib, seaborn | visualisation |
| scikit-learn | clustering, classification, pipelines |
| statsmodels, scipy | logistic regression, hypothesis / permutation tests |
| imbalanced-learn (`imblearn`) | class-imbalance resampling |
| kmodes | k-prototypes mixed-type clustering |
| shap | random-forest feature attribution |

### Run

1. Point `PATH` (first code cell) at your `Unified_Dataset_DG_cleaned.xlsx`.
2. Run cells top-to-bottom — Section 2 writes `firm_level_clean.csv`, which later sections load.

```bash
jupyter lab finals-2.ipynb
```

---

## Methodology notes

- **Leakage discipline** — targets, IDs, dates, geography, and post-grant outcomes are excluded from predictors; all transforms are per-row (no target leakage into scaling/imputation).
- **Imbalance-aware evaluation** — PR-AUC and minority-class recall over accuracy; class weights / resampling compared for sensitivity.
- **Small-sample honesty** — the equity model uses leave-one-out validation, bootstrap CIs, and permutation testing, and is explicitly reported as exploratory.
