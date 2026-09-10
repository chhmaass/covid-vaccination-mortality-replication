# Replication Code: Vaccination Intensity and Mortality During the COVID-19 Pandemic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18413113.svg)](https://doi.org/10.5281/zenodo.18413113)

This repository contains the **full replication code** for the manuscript:

> **Vaccination Intensity and Mortality During the COVID-19 Pandemic:  
> A Cross-National Panel Analysis with Lagged and Time-Varying Associations**  
> Christoph H. Maaß, PhD

The code in this repository reproduces **all tables and figures reported in the manuscript**, including robustness checks, using publicly available data.

---

## 1. Overview

The analysis evaluates population-level associations between COVID-19 vaccination intensity and mortality using a cross-national weekly panel (89 countries, March 2020 – July 2023). The replication code implements:

- Distributed vaccination lags (2–8 weeks)
- Placebo leads (2–4 weeks) to diagnose endogenous rollout
- Mixed-effects models with country random intercepts
- Explicit AR(1) residual correlation
- Fixed-effects and winsorized robustness specifications
- Separate analyses for:
  - All-cause mortality (ACM)
  - Non-COVID-19 mortality (NCM)

All results reported in the paper are reproducible from this repository.

---

## 2. Data Availability

### 2.1 Primary Analytic Dataset

The primary analytic dataset used in all models is the **Cross-National COVID-19 Risk and Policy Dataset (v1)**.

- **DOI-backed archive (recommended):**  
  https://doi.org/10.34740/kaggle/dsv/13637160

- **GitHub mirror (identical content):**  
  https://github.com/chhmaass/cross-national-covid-19-risk-policy-dataset-v1

The dataset includes harmonized weekly data on:

- All-cause mortality (STMF)
- COVID-19 deaths
- Vaccination intensity (Our World in Data)
- Epidemiological indicators
- Non-pharmaceutical interventions (OxCGRT)
- Viral variants
- Seasonality and weather controls

No proprietary or restricted data are used.

---

## 3. Repository Structure

```text
.
├─ README.md
├─ replication_vaccination_intensity_mortality.ipynb
├─ run_replication.py
├─ data/
│  └─ cross_national_covid_19_risk_policy_dataset_v1.csv   (not tracked; see Data Availability)
└─ output/
   ├─ tables/
   │  ├─ table_acm_ri_raw_full.csv
   │  ├─ table_ncm_ri_raw_full.csv
   │  ├─ table_acm_ri_raw_stmf.csv
   │  ├─ table_ncm_ri_raw_stmf.csv
   │  ├─ table_acm_ri_winsor.csv
   │  ├─ table_ncm_ri_winsor.csv
   │  ├─ table_acm_fe_raw.csv
   │  └─ table_ncm_fe_raw.csv
   └─ figures/
      ├─ figure_1_acm_ri_raw_full.png
      ├─ figure_2_ncm_ri_raw_full.png
      ├─ figure_3_acm_ri_raw_stmf.png
      ├─ figure_4_ncm_ri_raw_stmf.png
      ├─ figure_5_acm_ri_winsor.png
      ├─ figure_6_ncm_ri_winsor.png
      ├─ figure_7_acm_fe_raw.png
      └─ figure_8_ncm_fe_raw.png
```

### Description

- **`replication_vaccination_intensity_mortality.ipynb`**  
  Fully annotated notebook implementing the complete analysis pipeline.

- **`run_replication.py`**  
  Script version of the notebook enabling **fully automated, non-interactive replication** (e.g., for reviewers or batch execution).

- **`data/`**  
  Placeholder directory for the analytic dataset.  
  The dataset is **not version-controlled** and must be obtained from the sources listed in Section 2.

- **`output/`**  
  Automatically generated replication outputs:
  - `tables/` — all tables reported in the manuscript and appendix
  - `figures/` — all figures reported in the manuscript and appendix

All outputs are generated deterministically by running the notebook or script from top to bottom.

---

## 4. Software Requirements

The analysis was developed and tested with:

- **Python ≥ 3.10**
- Key Python packages:
  - pandas
  - numpy
  - statsmodels
  - linearmodels
  - matplotlib
  - seaborn
  - scipy
  - patsy

- **R ≥ 4.2**, with packages:
  - nlme
  - fixest

Exact package versions are not critical for qualitative replication; results are robust to minor version differences.

---

## 5. Replication Instructions

### Step 1: Obtain the data

Download the analytic dataset from the DOI-backed Kaggle archive **or** clone the GitHub mirror repository.  
Place the dataset in the local `data/` directory using the filename:

```text
cross_national_covid_19_risk_policy_dataset_v1.csv
```

### Step 2: Run the analysis

You may replicate the results using either interface.

**Option A: Jupyter Notebook**

```text
replication_vaccination_intensity_mortality.ipynb
```

Run all cells **from top to bottom**.  
The notebook is fully self-contained and does not rely on hidden state.

**Option B: Script (non-interactive)**

```bash
python run_replication.py
```

This reproduces all tables and figures without user interaction.

### Step 3: Generated outputs

Running the notebook or script reproduces:

- **Table 1:** Vaccination intensity and all-cause mortality (core specification)
- **Table 2:** Vaccination intensity and non-COVID-19 mortality (core specification)
- **Appendix Tables A1–A8:** Full robustness specifications
- **Figures 1–8:** Distributed lag, placebo lead, and robustness plots

All outputs are written to the `output/` directory.

---

## 6. Notes on Interpretation

- Results are population-level associations based on aggregate data.
- The analysis does not identify individual-level causal effects.
- Placebo leads are included explicitly to diagnose reactive vaccination rollout.
- Non-COVID-19 mortality is analyzed as a falsification outcome.

The replication code mirrors the manuscript’s identification strategy exactly.

---

## 7. Reproducibility and Transparency

All preprocessing, modeling, and robustness checks are implemented explicitly in code.  
There are **no manual steps**, **no hidden transformations**, and **no undocumented exclusions**.

Given identical input data, all numerical results are **deterministic and exactly reproducible**.

This repository fulfills the replication and transparency commitments stated in the manuscript.

---

## 8. Contact

For questions or replication issues:

**Christoph H. Maaß, PhD**  
Independent Researcher  
GitHub: https://github.com/chhmaass
