"""Data exploration of the raw applications file: the first thing to do BEFORE building any feature.

WHY: every modelling decision later (log transform, missing indicator, which columns to drop, which metric to use) is
justified by something you SEE here.  This script prints the facts the rest of the project builds on.

What it shows
  1. shape, dtypes, missing %            -> which columns need imputation?  (annual_income ~12%, bureau_score ~8%)
  2. class balance                       -> ~11% defaulters: accuracy is a useless metric, use AUC / KS / recall
  3. skew and outliers of income         -> log transform;  IQR rule flags the x10 data-entry errors
  4. is the missingness random?          -> MNAR-ish: self-employed / business skip annual_income more often
  5. correlation matrix + high pairs     -> keep ONE of each pair (|r| > 0.80)  [simple reasoning, no VIF]
  6. repeat applicants                   -> why we must split by time AND by applicant_id
  7. columns that must never be inputs   -> leakage / unavailable / sensitive (see docs/DATASET.md)

Formulas
  skewness  = mean(((x - mean)/std)^3)         0 = symmetric, > 1 = long right tail (income)
  IQR rule  : outlier if x > Q3 + 1.5*(Q3 - Q1)   Tiny example: Q1=2, Q3=6 -> upper fence = 6 + 1.5*4 = 12
  Pearson r = cov(x, y) / (std(x) * std(y))      +1 = move together, 0 = unrelated

Run:  python -m src.data.profile_data [--plots]       (--plots saves PNGs in reports/figures, needs matplotlib)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.paths import DATA_RAW, REPORTS
from src.features.definitions import LEAKAGE, SENSITIVE

CORR_THRESHOLD = 0.80
NUMERIC = ["age", "annual_income", "monthly_income", "employment_years", "loan_amount", "loan_term_months",
           "credit_history_years", "bureau_score", "existing_loans", "monthly_debt", "debt_to_income_ratio",
           "number_of_dependents"]


def iqr_outliers(s: pd.Series) -> tuple[int, float]:
    """Count values above Q3 + 1.5*IQR and return (count, fence)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    return int((s > fence).sum()), float(fence)


def profile(df: pd.DataFrame) -> dict:
    out: dict = {}
    print(f"1. SHAPE and TYPES: {df.shape[0]} applications x {df.shape[1]} columns")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(1), "n_unique": df.nunique()})
    print(info.to_string())
    out["shape"] = list(df.shape)
    out["missing_rates_pct"] = {k: float(v) for k, v in info["missing_%"].items() if v > 0}

    rate = float(df["default_12m"].mean())
    out["target_rate"] = round(rate, 4)
    print(f"\n2. CLASS BALANCE: default_12m = 1 for {rate:.1%} of applicants ({int(df.default_12m.sum())} of {len(df)}).")
    print(f"   A model that says 'nobody defaults' is {1 - rate:.1%} accurate and useless -> we use ROC-AUC, KS, PR-AUC and recall.")

    inc = df["annual_income"].dropna()
    n_out, fence = iqr_outliers(inc)
    out["income"] = dict(median=float(inc.median()), mean=float(inc.mean()), skew_raw=round(float(inc.skew()), 2),
                         skew_log=round(float(np.log1p(inc).skew()), 2), iqr_outliers=n_out, iqr_fence=round(fence))
    print(f"\n3. INCOME SHAPE: median {inc.median():,.0f}  mean {inc.mean():,.0f}  (mean > median = right tail)")
    print(f"   skewness raw = {inc.skew():.2f}   after log(1+x) = {np.log1p(inc).skew():.2f}   -> the log transform makes income symmetric")
    print(f"   IQR rule: fence = {fence:,.0f};  {n_out} values above it;  largest = {inc.max():,.0f} "
          f"({inc.max() / inc.median():.0f}x the median -> data-entry errors such as an extra zero)")
    q = df.assign(income_bad=df.annual_income > 5 * df.monthly_income * 12 * 1.0)["income_bad"].sum()
    out["income"]["reported_over_5x_monthly_x12"] = int(q)
    print(f"   rows where annual_income > 5 x (12 x monthly_income): {q}   (monthly_income is always present and noisy -> a sanity cross-check)")

    miss = df.assign(income_missing=df.annual_income.isna()).groupby("employment_type")["income_missing"].mean().round(3)
    hi = df.assign(income_missing=df.annual_income.isna(), high=df.monthly_income > df.monthly_income.median()).groupby("high")["income_missing"].mean().round(3)
    out["income_missing_by_employment"] = miss.to_dict()
    out["income_missing_by_high_earner"] = {str(k): v for k, v in hi.items()}
    print(f"\n4. MISSINGNESS of annual_income is NOT random (looks MNAR: depends on the value itself / type of work)")
    print("   share missing by employment_type:", miss.to_dict())
    print("   share missing for below-median vs above-median monthly_income:", {("high earners" if k else "others"): v for k, v in hi.items()})
    print("   -> median imputation + an explicit `income_missing` flag lets the model learn from the fact that income is missing.")

    corr = df[NUMERIC].corr().round(2)
    rank = df[NUMERIC].corr(method="spearman").round(2)          # rank correlation: robust to the x10 outliers
    pairs = [(a, b, float(corr.loc[a, b]), float(rank.loc[a, b])) for i, a in enumerate(NUMERIC) for b in NUMERIC[i + 1:]
             if max(abs(corr.loc[a, b]), abs(rank.loc[a, b])) >= 0.5]
    pairs.sort(key=lambda t: -max(abs(t[2]), abs(t[3])))
    out["top_correlations"] = [dict(a=a, b=b, pearson=r, spearman=rs) for a, b, r, rs in pairs[:8]]
    print(f"\n5. CORRELATION of numeric columns.  Pairs with |r| >= 0.5 (Pearson = linear, Spearman = rank-based, robust to outliers):")
    for a, b, r, rs in pairs[:8]:
        flag = "  <- above threshold: keep only ONE" if max(abs(r), abs(rs)) > CORR_THRESHOLD else ""
        print(f"   {a:>22s} ~ {b:<22s} Pearson {r:+.2f}  Spearman {rs:+.2f}{flag}")
    print("   Note: the few x10 income outliers pull Pearson(annual_income, monthly_income) down; the rank (Spearman) view shows the true strength.")
    print(f"   Rule used later (select_features.py): if |Spearman r| > {CORR_THRESHOLD}, keep the column that is more directly useful / easier to explain.")

    n_dup = int(df.applicant_id.duplicated().sum())
    out["repeat_applications"] = n_dup
    print(f"\n6. REPEAT APPLICANTS: {n_dup} applications ({n_dup / len(df):.1%}) share an applicant_id with an earlier one "
          "-> a random row split would put the same person in train AND test.")

    present = [c for c in LEAKAGE + SENSITIVE + ["age"] if c in df.columns]
    print(f"\n7. COLUMNS THAT MUST NOT BE MODEL INPUTS: {present}")
    print("   (leakage = information from after the decision; sensitive/restricted = policy; see docs/DATASET.md)")
    lk = {}
    for c in LEAKAGE:
        if pd.api.types.is_numeric_dtype(df[c]):
            lk[c] = round(float(df[c].corr(df.default_12m)), 2)                                   # Pearson r with the target
        else:                                                                                      # category: spread of default rate between groups
            rates = df.groupby(c)["default_12m"].mean()
            lk[c] = f"default rate {rates.min():.2f} .. {rates.max():.2f} across categories"
    out["leakage_corr_with_target"] = lk
    print("   correlation with the target (a leaky column looks 'too good'):", lk)
    return out


def plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = REPORTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    df.annual_income.dropna().hist(bins=60, ax=ax[0]); ax[0].set_title("annual_income (right-skewed)")
    np.log1p(df.annual_income.dropna()).hist(bins=60, ax=ax[1]); ax[1].set_title("log(1 + annual_income)")
    fig.tight_layout(); fig.savefig(figdir / "income_log.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    c = df[NUMERIC].corr()
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=90, fontsize=7); ax.set_yticks(range(len(NUMERIC)), NUMERIC, fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(figdir / "correlation.png", dpi=110); plt.close(fig)
    print(f"figures saved in {figdir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(DATA_RAW / "applications.csv")
    out = profile(df)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_profile.json").write_text(json.dumps(out, indent=2))
    if a.plots:
        plots(df)


if __name__ == "__main__":
    main()
