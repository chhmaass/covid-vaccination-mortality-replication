#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replication Pipeline (publication-ready)
=======================================

Auto-converted and refactored from:
  replication_vaccination_intensity_mortality.ipynb

This script reproduces all main and robustness analyses reported in:

  Christoph H. Maass (2025). Vaccination Intensity and Mortality During the COVID-19 Pandemic.

Key improvements vs. the raw notebook export
--------------------------------------------
✅ Single-source configuration (no duplicated lists/dicts)
✅ Reusable preprocessing & model runners (RI/FE; ACM/NCM; full/STMF; raw/winsorized)
✅ Deterministic, auditable outputs:
   - output/tables/*.csv  (Excel-safe UTF-8-SIG)
   - output/figures/*.png
   - output/run_metadata.json (versions, paths, run time, sample sizes)
✅ Stronger failure modes:
   - input validation (required columns, non-empty samples)
   - consistent analytic sample handling
   - explicit R package checks (no installs)
✅ Clean separation between:
   - data transformations (Python)
   - estimation/inference (R via rpy2)

How to run
----------
1) Ensure the dataset exists at: ./data/cross_national_covid_19_risk_policy_dataset_v1.csv
2) Ensure R packages are installed: nlme, fixest
3) Run:
   python replication_vaccination_intensity_mortality.py

Notes
-----
- This script does NOT install anything (publication safety).
- Figures/tables are deterministic given the input CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys
import platform
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri


# ============================================================
# 0) Global configuration (single source of truth)
# ============================================================

# Lags/leads for distributed-lag + placebo-lead specification
LAGS: List[int] = [2, 4, 6, 8]
LEADS: List[int] = [2, 4]

# Mapping: coefficient name -> human-readable label (used in ALL tables)
TERM_LABELS: Dict[str, str] = {
    "vax_lead2": "Vaccination lead (t+2)",
    "vax_lead4": "Vaccination lead (t+4)",
    "vax_lag2": "Vaccination lag (t-2)",
    "vax_lag4": "Vaccination lag (t-4)",
    "vax_lag6": "Vaccination lag (t-6)",
    "vax_lag8": "Vaccination lag (t-8)",
    "vax_lag6_x_tc": "Lag 6 × calendar time",
    "vax_lag8_x_tc": "Lag 8 × calendar time",
}

# Mapping: coefficient name -> event time (weeks relative to vaccination)
TERM_TO_TIME: Dict[str, int] = {
    "vax_lead4": -4,
    "vax_lead2": -2,
    "vax_lag2": +2,
    "vax_lag4": +4,
    "vax_lag6": +6,
    "vax_lag8": +8,
}

# STMF subsample (as used in your script)
STMF_COUNTRIES: List[str] = [
    "AUS", "AUT", "BEL", "BGR", "CAN", "CHE", "CHL", "CZE", "DEU", "DNK",
    "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN", "ISL", "ISR",
    "ITA", "KOR", "LTU", "LUX", "LVA", "NLD", "NOR", "NZL", "POL", "PRT",
    "RUS", "SVK", "SVN", "SWE", "TWN", "USA"
]

# Required base columns in the CSV (hard fail if missing)
REQUIRED_COLUMNS: List[str] = [
    "date",
    "country_iso3",
    "all_cause_mortality",
    "covid_19_mortality_original",
    "covid_19_vaccine_doses_administered",
    "covid_19_incidence_original",
    "covid_19_prevalence_original",
    "covid_19_policy_stringency",
    "covid_19_face_covering_policy",
    "covid_19_testing_tracing_policy",
    "covid_19_variant_delta",
    "covid_19_variant_omicron",
    "sine_seasonality",
    "cosine_seasonality",
    "temperature",
    "relative_humidity",
]

# Covariates included in all models (besides the outcome and vaccination terms)
BASE_COVARS: List[str] = [
    "t_c",
    "covid_19_incidence_original",
    "covid_19_prevalence_original",
    "covid_19_policy_stringency",
    "covid_19_face_covering_policy",
    "covid_19_testing_tracing_policy",
    "covid_19_variant_delta",
    "covid_19_variant_omicron",
    "sine_seasonality",
    "cosine_seasonality",
    "temperature",
    "relative_humidity",
]

# Vaccination terms included in all models
VAX_TERMS: List[str] = [
    "vax_lead2", "vax_lead4",
    "vax_lag2", "vax_lag4", "vax_lag6", "vax_lag8",
    "vax_lag6_x_tc", "vax_lag8_x_tc",
]

# Excel-proof encoding for reviewer convenience
CSV_ENCODING = "utf-8-sig"


# ============================================================
# 1) Paths & I/O
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cross_national_covid_19_risk_policy_dataset_v1.csv"


# ============================================================
# 2) Utilities (validation, logging, winsorization, formatting)
# ============================================================

def die(msg: str, code: int = 1) -> None:
    """Fail fast with a clear message (publication-grade behavior)."""
    raise SystemExit(f"\n[ERROR] {msg}\n")


def check_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"Missing required columns in CSV: {missing}")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    check_required_columns(df, REQUIRED_COLUMNS)
    return df


def winsorize_by_country(
    df: pd.DataFrame,
    country_col: str,
    value_col: str,
    lo: float = 0.01,
    hi: float = 0.99
) -> pd.Series:
    """
    Winsorize a column within-country using quantiles.
    NOTE: publication risk: can change estimand; keep as robustness only.
    """
    qs = (
        df.groupby(country_col)[value_col]
        .quantile([lo, hi])
        .unstack()
        .rename(columns={lo: "lo", hi: "hi"})
    )
    lo_map = df[country_col].map(qs["lo"])
    hi_map = df[country_col].map(qs["hi"])
    return df[value_col].clip(lower=lo_map, upper=hi_map)


def format_vax_table(tt: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize model output to a reviewer-friendly vaccination-only table.
    Accepts either nlme tTable style or fixest coeftable style.
    """
    # nlme: columns are Value, Std.Error, p-value
    if {"Value", "Std.Error", "p-value", "term"}.issubset(tt.columns):
        est_col, se_col, p_col = "Value", "Std.Error", "p-value"
    # fixest: estimate, se, p_value
    elif {"estimate", "se", "p_value", "term"}.issubset(tt.columns):
        est_col, se_col, p_col = "estimate", "se", "p_value"
    else:
        die(
            f"Unexpected coefficient table format. Columns: {list(tt.columns)}")

    keep = [t for t in VAX_TERMS if t in tt["term"].values]
    out = tt[tt["term"].isin(keep)].copy()
    out["term"] = pd.Categorical(
        out["term"], categories=VAX_TERMS, ordered=True)
    out = out.sort_values("term")

    out["Term"] = out["term"].map(TERM_LABELS)
    out["Estimate"] = out[est_col].map(lambda x: f"{x:+.3f}")
    out["SE"] = out[se_col].map(lambda x: f"{x:.3f}")
    out["p-value"] = out[p_col].map(lambda x: f"{x:.3f}")

    return out[["Term", "Estimate", "SE", "p-value"]]


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def plot_distributed_lag(
    tt: pd.DataFrame,
    title: str,
    ylabel: str,
    outpath: Path
) -> None:
    """
    Plot distributed-lag coefficients with 95% CI.
    Works for nlme tt (Value/Std.Error) and fixest tt (estimate/se).
    """
    if {"Value", "Std.Error", "term"}.issubset(tt.columns):
        est_col, se_col = "Value", "Std.Error"
    elif {"estimate", "se", "term"}.issubset(tt.columns):
        est_col, se_col = "estimate", "se"
    else:
        die(
            f"Unexpected coefficient table format for plotting. Columns: {list(tt.columns)}")

    dfp = tt[tt["term"].isin(TERM_TO_TIME.keys())].copy()
    if dfp.empty:
        die("No vaccination lead/lag terms found for plot.")

    dfp["time"] = dfp["term"].map(TERM_TO_TIME)
    dfp = dfp.sort_values("time")

    x = dfp["time"].to_numpy()
    y = dfp[est_col].to_numpy()
    se = dfp[se_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        x, y,
        yerr=1.96 * se,
        fmt="o",
        capsize=4
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:+d}" for t in x])
    ax.set_xlabel(
        "Weeks relative to vaccination\n(Negative = placebo leads, Positive = post-vaccination lags)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 3) Data preparation (single implementation)
# ============================================================

@dataclass(frozen=True)
class SampleSpec:
    outcome: str  # "ACM" or "NCM"
    sample: str   # "full" or "stmf"
    outlier: str  # "raw" or "winsor"


def prepare_dataframe(
    df_raw: pd.DataFrame,
    spec: SampleSpec
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare the analytic dataframe for a given outcome/sample/outlier setting.
    Returns (df_m, outcome_column_name).
    """
    df = df_raw.copy()

    # Standard country key
    df["country"] = df["country_iso3"].astype(str)

    # Restrict sample if needed
    if spec.sample == "stmf":
        df = df[df["country"].isin(STMF_COUNTRIES)].copy()
    elif spec.sample != "full":
        die(f"Unknown sample: {spec.sample}")

    # Create outcome column
    if spec.outcome == "ACM":
        ycol = "all_cause_mortality"
    elif spec.outcome == "NCM":
        ycol = "non_covid_mortality"
        df[ycol] = df["all_cause_mortality"] - \
            df["covid_19_mortality_original"]
    else:
        die(f"Unknown outcome: {spec.outcome}")

    # Outlier handling (robustness)
    if spec.outlier == "winsor":
        df[ycol] = winsorize_by_country(df, "country", ycol, lo=0.01, hi=0.99)
    elif spec.outlier != "raw":
        die(f"Unknown outlier setting: {spec.outlier}")

    # Sort and build panel time index
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    df["t"] = df.groupby("country").cumcount()
    df["t_c"] = df["t"] - df.groupby("country")["t"].transform("mean")

    # Vaccination lags/leads
    for k in LAGS:
        df[f"vax_lag{k}"] = df.groupby(
            "country")["covid_19_vaccine_doses_administered"].shift(k)
    for k in LEADS:
        df[f"vax_lead{k}"] = df.groupby(
            "country")["covid_19_vaccine_doses_administered"].shift(-k)

    # Interactions
    df["vax_lag6_x_tc"] = df["vax_lag6"] * df["t_c"]
    df["vax_lag8_x_tc"] = df["vax_lag8"] * df["t_c"]

    # Analytic sample
    model_vars = [ycol, "country", "t", "t_c"] + VAX_TERMS + \
        BASE_COVARS[1:]  # BASE_COVARS includes t_c; avoid dup
    df_m = df.dropna(subset=model_vars).copy()

    if df_m.empty:
        die(f"Analytic sample is empty after NA dropping for spec={spec}.")

    return df_m, ycol


# ============================================================
# 4) R integration (single implementations)
# ============================================================

def r_dependency_check() -> None:
    ro.r(r"""
    required_pkgs <- c("nlme", "fixest")

    missing_pkgs <- required_pkgs[!vapply(
      required_pkgs,
      requireNamespace,
      FUN.VALUE = logical(1),
      quietly = TRUE
    )]

    if (length(missing_pkgs) > 0) {
      stop(
        paste0(
          "Missing required R packages: ",
          paste(missing_pkgs, collapse = ", "),
          "\nPlease install them in R, e.g.:\n",
          "install.packages(c(",
          paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
          "))"
        ),
        call. = FALSE
      )
    }
    """)


def run_ri_model(df_m: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """
    Random-intercept model with AR(1) residuals:
      nlme::lme(... random = ~1|country, correlation=corAR1(~t|country))
    Returns coefficient table with: term, Value, Std.Error, p-value.
    """
    with localconverter(pandas2ri.converter):
        ro.globalenv["df_m"] = df_m

    ro.r(f"""
    suppressMessages(library(nlme))
    df_m$country <- as.factor(df_m$country)

    m_core <- nlme::lme(
      fixed = {ycol} ~
        t_c +
        vax_lead2 + vax_lead4 +
        vax_lag2 + vax_lag4 + vax_lag6 + vax_lag8 +
        vax_lag6_x_tc + vax_lag8_x_tc +
        covid_19_incidence_original +
        covid_19_prevalence_original +
        covid_19_policy_stringency +
        covid_19_face_covering_policy +
        covid_19_testing_tracing_policy +
        covid_19_variant_delta +
        covid_19_variant_omicron +
        sine_seasonality + cosine_seasonality +
        temperature + relative_humidity,
      random = ~ 1 | country,
      correlation = corAR1(form = ~ t | country),
      data = df_m,
      method = "REML",
      control = nlme::lmeControl(
        opt = "optim",
        msMaxIter = 200,
        niterEM = 50
      )
    )

    tt <- as.data.frame(summary(m_core)$tTable)
    tt$term <- rownames(tt)
    rownames(tt) <- NULL
    tt
    """)

    with localconverter(pandas2ri.converter):
        tt = ro.r("tt")

    return tt


def run_fe_model(df_m: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """
    Fixed-effects (country) model using fixest::feols with Newey–West (nlag=1).
    Returns coefficient table with: term, estimate, se, p_value.
    """
    with localconverter(pandas2ri.converter):
        ro.globalenv["df_m"] = df_m

    ro.r(f"""
    suppressMessages(library(fixest))
    df_m$country <- as.factor(df_m$country)

    m_core_fe <- feols(
      {ycol} ~
        t_c +
        vax_lead2 + vax_lead4 +
        vax_lag2 + vax_lag4 + vax_lag6 + vax_lag8 +
        vax_lag6_x_tc + vax_lag8_x_tc +
        covid_19_incidence_original +
        covid_19_prevalence_original +
        covid_19_policy_stringency +
        covid_19_face_covering_policy +
        covid_19_testing_tracing_policy +
        covid_19_variant_delta +
        covid_19_variant_omicron +
        sine_seasonality + cosine_seasonality +
        temperature + relative_humidity
      | country,
      data = df_m,
      panel.id = ~country + t
    )

    tt <- as.data.frame(coeftable(m_core_fe, vcov = "NW", nlag = 1))
    tt$term <- rownames(tt)
    rownames(tt) <- NULL

    names(tt) <- sub("^Estimate$", "estimate", names(tt))
    names(tt) <- sub("^Std\\\\. Error$", "se", names(tt))
    names(tt) <- sub("^Pr\\\\(>\\\\|t\\\\|\\\\)$", "p_value", names(tt))
    tt
    """)

    with localconverter(pandas2ri.converter):
        tt = ro.r("tt")

    return tt


# ============================================================
# 5) Orchestration (all models, consistent naming)
# ============================================================

def describe_sample(df_m: pd.DataFrame) -> Dict[str, int]:
    return {"countries": int(df_m["country"].nunique()), "rows": int(df_m.shape[0])}


def run_and_save(
    df_raw: pd.DataFrame,
    spec: SampleSpec,
    model_family: str,  # "RI" or "FE"
    table_name: str,
    fig_name: str,
    fig_title: str,
    fig_ylabel: str
) -> Dict[str, object]:
    """
    Prepare data, run model, save vaccination-only table + distributed-lag plot.
    Returns metadata for run log.
    """
    df_m, ycol = prepare_dataframe(df_raw, spec)
    sample_info = describe_sample(df_m)

    # Model estimation
    if model_family == "RI":
        tt = run_ri_model(df_m, ycol)
    elif model_family == "FE":
        tt = run_fe_model(df_m, ycol)
    else:
        die(f"Unknown model family: {model_family}")

    # Outputs
    tab = format_vax_table(tt)
    save_table(tab, TABLE_DIR / table_name)
    plot_distributed_lag(
        tt=tt,
        title=fig_title,
        ylabel=fig_ylabel,
        outpath=FIGURE_DIR / fig_name
    )

    # Console summary (kept concise for reproducibility logs)
    print(
        f"\n=== {spec.outcome} — {model_family} — {spec.outlier.upper()} — {spec.sample.upper()} ===")
    print(
        f"Countries: {sample_info['countries']}, Rows: {sample_info['rows']}")
    print(tab.to_string(index=False))

    return {
        "spec": spec.__dict__,
        "model_family": model_family,
        "outcome_column": ycol,
        "sample_info": sample_info,
        "table": str((TABLE_DIR / table_name).relative_to(BASE_DIR)),
        "figure": str((FIGURE_DIR / fig_name).relative_to(BASE_DIR)),
    }


def write_run_metadata(metadata: Dict[str, object]) -> None:
    out = OUTPUT_DIR / "run_metadata.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================
# 6) Main
# ============================================================

def main() -> None:
    # R package check (hard fail early)
    r_dependency_check()

    # Load data once
    df_raw = safe_read_csv(DATA_PATH)

    # Run meta
    run_started = dt.datetime.now(dt.timezone.utc).isoformat()
    meta: Dict[str, object] = {
        "run_started_utc": run_started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_path": str(DATA_PATH.relative_to(BASE_DIR)) if DATA_PATH.exists() else str(DATA_PATH),
        "outputs": {
            "tables_dir": str(TABLE_DIR.relative_to(BASE_DIR)),
            "figures_dir": str(FIGURE_DIR.relative_to(BASE_DIR)),
        },
        "models": [],
    }

    # -------------------------
    # A) Random-intercept (RI) models
    # -------------------------
    # 1) ACM — RI — RAW — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="ACM", sample="full", outlier="raw"),
        model_family="RI",
        table_name="table_acm_ri_raw_full.csv",
        fig_name="figure_1_acm_ri_raw_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and All-Cause Mortality (RI, RAW, full)",
        fig_ylabel="Effect on all-cause mortality\n(deaths per million per day)",
    ))

    # 2) NCM — RI — RAW — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="NCM", sample="full", outlier="raw"),
        model_family="RI",
        table_name="table_ncm_ri_raw_full.csv",
        fig_name="figure_2_ncm_ri_raw_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and Non-COVID-19 Mortality (RI, RAW, full)",
        fig_ylabel="Effect on non-COVID-19 mortality\n(deaths per million per day)",
    ))

    # 3) ACM — RI — RAW — STMF
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="ACM", sample="stmf", outlier="raw"),
        model_family="RI",
        table_name="table_acm_ri_raw_stmf.csv",
        fig_name="figure_3_acm_ri_raw_stmf.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and All-Cause Mortality (RI, RAW, STMF)",
        fig_ylabel="Effect on all-cause mortality\n(deaths per million per day)",
    ))

    # 4) NCM — RI — RAW — STMF
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="NCM", sample="stmf", outlier="raw"),
        model_family="RI",
        table_name="table_ncm_ri_raw_stmf.csv",
        fig_name="figure_4_ncm_ri_raw_stmf.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and Non-COVID-19 Mortality (RI, RAW, STMF)",
        fig_ylabel="Effect on non-COVID-19 mortality\n(deaths per million per day)",
    ))

    # 5) ACM — RI — WINSORIZED — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="ACM", sample="full", outlier="winsor"),
        model_family="RI",
        table_name="table_acm_ri_winsor_full.csv",
        fig_name="figure_5_acm_ri_winsor_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and All-Cause Mortality (RI, WINSORIZED, full)",
        fig_ylabel="Effect on all-cause mortality\n(deaths per million per day)",
    ))

    # 6) NCM — RI — WINSORIZED — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="NCM", sample="full", outlier="winsor"),
        model_family="RI",
        table_name="table_ncm_ri_winsor_full.csv",
        fig_name="figure_6_ncm_ri_winsor_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and Non-COVID-19 Mortality (RI, WINSORIZED, full)",
        fig_ylabel="Effect on non-COVID-19 mortality\n(deaths per million per day)",
    ))

    # -------------------------
    # B) Fixed-effects (FE) models
    # -------------------------
    # 7) ACM — FE — RAW — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="ACM", sample="full", outlier="raw"),
        model_family="FE",
        table_name="table_acm_fe_raw_full.csv",
        fig_name="figure_7_acm_fe_raw_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and All-Cause Mortality (FE, RAW, full)",
        fig_ylabel="Effect on all-cause mortality",
    ))

    # 8) NCM — FE — RAW — full sample
    meta["models"].append(run_and_save(
        df_raw=df_raw,
        spec=SampleSpec(outcome="NCM", sample="full", outlier="raw"),
        model_family="FE",
        table_name="table_ncm_fe_raw_full.csv",
        fig_name="figure_8_ncm_fe_raw_full.png",
        fig_title="Distributed Lag and Placebo Lead Associations\nVaccination Intensity and Non-COVID-19 Mortality (FE, RAW, full)",
        fig_ylabel="Effect on non-COVID-19 mortality",
    ))

    # Finalize metadata
    meta["run_finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_run_metadata(meta)

    print("\n✅ Finished. Outputs written to:")
    print(f"  - {TABLE_DIR.relative_to(BASE_DIR)}")
    print(f"  - {FIGURE_DIR.relative_to(BASE_DIR)}")
    print(f"  - { (OUTPUT_DIR / 'run_metadata.json').relative_to(BASE_DIR) }")


if __name__ == "__main__":
    main()
