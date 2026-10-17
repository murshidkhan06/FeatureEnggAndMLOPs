"""Synthetic retail-bank loan applications (100% generated, seeded, no downloads).

WHY synthetic?  Real credit data is confidential and regulated.  A generator lets us (a) know the
TRUE rules that create defaults, (b) inject the real-world problems students must learn to spot,
and (c) create "production" batches that drift on purpose.

Business story
--------------
A bank decides, at APPLICATION time, whether to approve an unsecured personal loan.
Target  default_12m = 1 if the borrower defaults within 12 months of disbursal, else 0 (~10% are 1).

Real-world problems injected (all documented in docs/DATASET.md)
----------------------------------------------------------------
* annual_income is missing for ~12% of rows, and NOT at random (MNAR-ish): self-employed / business
  people and high earners skip it more often.  monthly_income is always present but noisy.
* annual_income and monthly_income are highly correlated (r ~ 0.9); monthly_debt and
  debt_to_income_ratio are correlated too.
* income is right-skewed with a few data-entry outliers (x10) -> log transform.
* class imbalance (~10% defaults).
* leakage / not-available-at-application-time columns (interest_rate_offered, loan_approved,
  days_past_due_after_disbursal, collections_status, post_disbursal_payment_ratio,
  credit_officer_override): they contain information from the FUTURE or from the decision itself.
* sensitive attributes (gender, marital_status) and a restricted one (age); `location` (15 cities)
  can act as a PROXY for gender in this data (some cities have more women applicants AND a local
  economic effect on default).
* ~7% of applications are RE-APPLICATIONS by an earlier applicant (same applicant_id, nearly the same
  features, same outcome) -> a random row split would put "the same person" in train and test.

Worked example of the default rule (what the generator uses, simplified)
-----------------------------------------------------------------------
    logit = b0 + 0.55*z_dti + 0.45*z_loan_to_income - 0.45*latent_creditworthiness + ...
    p     = 1 / (1 + exp(-logit))          e.g. logit = -2.2  ->  p = 1/(1+e^2.2) = 0.10
    default_12m = 1 if U < p else 0        (U is a uniform random number in [0, 1))

Run:  python -m src.data.make_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import DATA_PRODUCTION, DATA_RAW

SEED = 42
START, END = "2022-01-01", "2023-12-31"          # training era (labels are mature: 12m passed)

# city -> (tier, share of FEMALE applicants in that city, local economy effect on default logit)
# The female-share and the local effect are deliberately correlated: that is what makes
# `location` a PROXY for gender in this synthetic world.
CITIES = {
    "Mumbai": (1, 0.30, -0.20), "Delhi": (1, 0.32, -0.10), "Bengaluru": (1, 0.28, -0.25),
    "Chennai": (1, 0.45, 0.15), "Hyderabad": (1, 0.48, 0.20),
    "Jaipur": (2, 0.30, -0.10), "Lucknow": (2, 0.50, 0.25), "Indore": (2, 0.33, -0.05),
    "Coimbatore": (2, 0.52, 0.20), "Nagpur": (2, 0.35, 0.00),
    "Gorakhpur": (3, 0.55, 0.35), "Bhilai": (3, 0.30, -0.05), "Hubballi": (3, 0.52, 0.30),
    "Siliguri": (3, 0.34, 0.00), "Bhagalpur": (3, 0.58, 0.40),
}
CITY_WEIGHTS = np.array([14, 14, 12, 8, 8, 5, 5, 5, 4, 4, 3, 3, 3, 3, 3], dtype=float)
EMPLOYMENT_TYPES = ["salaried", "self_employed", "gig", "business"]
EDUCATION_LEVELS = ["school", "diploma", "graduate", "postgraduate"]

B0 = -2.62   # intercept chosen so that the training data has ~10% defaults

# "Scenarios" change the world the model sees.  BASE = training era.
BASE = dict(income_scale=1.0, dti_scale=1.0, bureau_shift=0.0, hist_shift=0.0, intercept_shift=0.0,
            bureau_coef_scale=1.0, dti_coef_scale=1.0, gig_extra=0.0, self_extra=0.0, loan_term_extra=0.0,
            shock_sd=0.0)   # shock_sd: unpredictable life shocks (job loss, illness) -> noise the model cannot see -> lower AUC
NORMAL_PROD = {**BASE, "hist_shift": 0.45}       # applicants slowly get longer credit histories: "significant but small"
# Economic downturn = covariate shift (income down, DTI up, bureau scores lower) AND concept drift (the old rules weaken:
# bureau score predicts less, gig / self-employed are hit harder, unpredictable shocks hit everybody).
DOWNTURN = {**BASE, "income_scale": 0.82, "dti_scale": 1.30, "bureau_shift": -22.0, "hist_shift": 0.45,
            "intercept_shift": 0.0, "bureau_coef_scale": 0.30, "dti_coef_scale": 1.0,
            "gig_extra": 0.9, "self_extra": 0.5, "loan_term_extra": 0.012, "shock_sd": 1.4}


def _draw_applicants(rng: np.random.Generator, n: int, sc: dict, start: str, end: str) -> pd.DataFrame:
    """Draw applicant attributes (who applies) - no outcome yet."""
    cities = list(CITIES)
    city = rng.choice(cities, n, p=CITY_WEIGHTS / CITY_WEIGHTS.sum())
    tier = np.array([CITIES[c][0] for c in city])
    female_share = np.array([CITIES[c][1] for c in city])
    gender = np.where(rng.random(n) < female_share, "female", "male")
    age = np.clip(rng.normal(35, 9, n), 21, 65).round().astype(int)
    emp_p = np.array([0.55, 0.18, 0.12, 0.15])
    employment = rng.choice(EMPLOYMENT_TYPES, n, p=emp_p)
    # slightly more gig work among younger applicants and among women (occupational mix)
    gig_flip = (rng.random(n) < 0.06 * (gender == "female") + 0.05 * (age < 28)) & (employment == "salaried")
    employment = np.where(gig_flip, "gig", employment)
    education = rng.choice(EDUCATION_LEVELS, n, p=[0.15, 0.20, 0.45, 0.20])
    edu_idx = pd.Series(education).map({e: i for i, e in enumerate(EDUCATION_LEVELS)}).to_numpy()
    marital = np.where(age < 27, rng.choice(["single", "married"], n, p=[0.8, 0.2]),
                       rng.choice(["married", "single", "divorced"], n, p=[0.68, 0.22, 0.10]))

    emp_eff = pd.Series(employment).map({"salaried": 0.0, "self_employed": 0.15, "gig": -0.30, "business": 0.35}).to_numpy()
    log_inc = (np.log(480_000) + 0.12 * edu_idx + emp_eff + 0.012 * (age - 35) + 0.08 * (tier == 1)
               - 0.12 * (gender == "female") + rng.normal(0, 0.34, n))
    true_annual = np.exp(log_inc) * sc["income_scale"]
    lat = rng.normal(0, 1, n)                       # unobserved "creditworthiness"

    emp_years = np.minimum(rng.gamma(2.0, 3.2, n), age - 19.0)
    hist = np.minimum(rng.gamma(2.4, 2.9, n) + sc["hist_shift"], age - 18.0).clip(0, None)
    hist = np.where(rng.random(n) < 0.07, rng.uniform(0, 0.9, n), hist)     # a few thin-file applicants
    existing = np.clip(rng.poisson(0.8 + 0.25 * (log_inc - np.log(480_000)) + 0.04 * hist), 0, 6)
    dti_true = np.clip(np.exp(rng.normal(np.log(0.20), 0.70, n)) * (1 + 0.12 * existing) * sc["dti_scale"], 0.01, 0.90)
    true_monthly = true_annual / 12
    monthly_income = (true_monthly * np.exp(rng.normal(0, 0.20, n))).round(-2)        # noisy "last 3 months" income
    monthly_debt = (dti_true * true_monthly).round(-2)
    loan_amount = np.clip(true_annual * np.exp(rng.normal(np.log(0.50), 0.45, n)), 30_000, 2_500_000).round(-3)
    term = rng.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.25, 0.30, 0.20, 0.15])
    dependents = np.clip(rng.poisson(0.9 + 0.5 * (marital == "married")), 0, 5)
    bureau = np.clip(650 + 60 * lat + 4 * np.minimum(hist, 10) - 0.0 * existing + rng.normal(0, 28, n)
                     + sc["bureau_shift"], 300, 900).round()
    bureau_missing = rng.random(n) < np.where(hist < 1.0, 0.7, 0.02)               # thin file -> no bureau score
    bureau = np.where(bureau_missing, np.nan, bureau)

    # MNAR-ish missingness of the *reported* annual income
    z_inc = (log_inc - np.log(480_000)) / 0.5
    p_miss = 1 / (1 + np.exp(-(-2.75 + 0.9 * np.isin(employment, ["self_employed", "business"]) + 0.55 * z_inc)))
    income_missing = rng.random(n) < p_miss
    reported = (true_annual.round(-3)).astype(float)
    outlier = rng.random(n) < 0.002                                             # data-entry error: x10
    reported = np.where(outlier, reported * 10, reported)
    reported = np.where(income_missing, np.nan, reported)

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days + 1, n), unit="D")
    return pd.DataFrame(dict(
        application_date=dates, age=age, annual_income=reported, monthly_income=monthly_income,
        employment_years=emp_years.round(1), loan_amount=loan_amount, loan_term_months=term,
        credit_history_years=hist.round(1), bureau_score=bureau, existing_loans=existing, monthly_debt=monthly_debt,
        number_of_dependents=dependents, employment_type=employment, education=education, location=city,
        gender=gender, marital_status=marital,
        _lat=lat, _true_annual=true_annual, _dti_true=dti_true, _U=rng.random(n), _noise=rng.random(n)))


def _apply_outcomes(df: pd.DataFrame, rng: np.random.Generator, sc: dict) -> pd.DataFrame:
    """Turn attributes into default probability, the 0/1 target and the leakage columns."""
    z_dti = (df["_dti_true"] - 0.25) / 0.15
    lti_true = df["loan_amount"] / df["_true_annual"]
    z_lti = (lti_true - 0.55) / 0.35
    z_inc = (np.log(df["_true_annual"]) - np.log(480_000)) / 0.5
    emp = df["employment_type"]
    logit = (B0 + sc["intercept_shift"]
             + 0.50 * sc["dti_coef_scale"] * z_dti + 0.80 * z_lti - 0.30 * z_inc
             - 0.45 * sc["bureau_coef_scale"] * df["_lat"]
             - 0.10 * np.minimum(df["credit_history_years"], 10) + 0.35 * (df["credit_history_years"] < 1)
             + 0.55 * np.exp(-df["employment_years"] / 1.5)
             + 0.18 * df["existing_loans"] + 0.08 * df["number_of_dependents"]
             + (0.004 + sc["loan_term_extra"]) * (df["loan_term_months"] - 36)
             + (emp == "gig") * (0.55 + sc["gig_extra"]) + (emp == "self_employed") * (0.30 + sc["self_extra"])
             + (emp == "business") * 0.15
             + np.where(df["age"] < 25, 0.45, np.where(df["age"] < 30, 0.20, 0.0))
             + df["location"].map({c: v[2] for c, v in CITIES.items()}))
    if sc["shock_sd"] > 0:                       # downturn only: shocks no application feature can predict
        logit = logit + rng.normal(0, sc["shock_sd"], len(df))
    p = 1 / (1 + np.exp(-logit))
    y = (df["_U"] < p).astype(int)
    out = df.copy()
    out["_p_true"] = p
    out["default_12m"] = y
    # ---- leakage / unavailable-at-application-time columns --------------------------------------
    # interest_rate_offered: set AFTER risk scoring (and re-priced after review) -> carries the answer
    out["interest_rate_offered"] = (11.0 + 25 * p + 3.2 * y + rng.normal(0, 1.1, len(df))).round(2)
    # loan_approved: the historical DECISION (an OUTPUT of the process, not an input)
    out["loan_approved"] = (rng.random(len(df)) < np.clip(0.97 - 0.9 * p - 0.15 * y, 0.05, 0.99)).astype(int)
    # days_past_due_after_disbursal: only known months AFTER the loan was paid out
    dpd = np.where(y == 1, rng.gamma(4, 18, len(df)), np.where(rng.random(len(df)) < 0.06, rng.integers(1, 15, len(df)), 0))
    out["days_past_due_after_disbursal"] = np.round(dpd).astype(int)
    # collections_status: created by the collections team after missed payments
    r = rng.random(len(df))
    coll = np.where(y == 1, np.where(r < 0.35, "reminder", np.where(r < 0.7, "legal", "written_off")),
                    np.where(r < 0.96, "none", "reminder"))
    out["collections_status"] = coll
    # post_disbursal_payment_ratio: share of due EMIs paid in the first 6 months (future information)
    out["post_disbursal_payment_ratio"] = np.clip(np.where(y == 1, rng.beta(4, 3, len(df)), rng.beta(40, 1.2, len(df))), 0, 1).round(3)
    # credit_officer_override: a human step AFTER the model score, with later knowledge
    ro = rng.random(len(df))
    out["credit_officer_override"] = np.where(y == 1, np.where(ro < 0.30, "downgrade", np.where(ro < 0.35, "upgrade", "none")),
                                              np.where(ro < 0.03, "downgrade", np.where(ro < 0.10, "upgrade", "none")))
    return out


def generate(n: int = 12_000, seed: int = SEED, scenario: dict | None = None, start: str = START, end: str = END,
             reapply_rate: float = 0.07, id_prefix: str = "A") -> pd.DataFrame:
    """Generate `n` loan applications.  Same seed -> identical data (deterministic)."""
    sc = {**BASE, **(scenario or {})}
    rng = np.random.default_rng(seed)
    n_re = int(round(n * reapply_rate))
    base = _draw_applicants(rng, n - n_re, sc, start, end)
    base["applicant_id"] = [f"{id_prefix}{i:06d}" for i in range(1, len(base) + 1)]

    # --- re-applications: an earlier applicant comes back later with almost the same profile ----------
    src = base.sample(n_re, random_state=seed, replace=False).copy() if n_re else base.iloc[:0].copy()
    if n_re:
        gap = pd.to_timedelta(rng.integers(45, 330, n_re), unit="D")
        src["application_date"] = (src["application_date"] + gap).clip(upper=pd.Timestamp(end))
        for col, rel in [("annual_income", 0.03), ("monthly_income", 0.03), ("loan_amount", 0.05), ("monthly_debt", 0.05)]:
            src[col] = (src[col] * np.exp(rng.normal(0, rel, n_re))).round(-2 if col != "loan_amount" else -3)
        src["age"] = src["age"] + (gap.days > 180).astype(int)
        src["credit_history_years"] = (src["credit_history_years"] + gap.days / 365).round(1)
        src["employment_years"] = (src["employment_years"] + gap.days / 365).round(1)
        src["_noise"] = rng.random(n_re)
    df = pd.concat([base, src], ignore_index=True)
    df = _apply_outcomes(df, rng, sc)
    df = df.sort_values(["application_date", "applicant_id"], kind="stable").reset_index(drop=True)
    df["monthly_income"] = df["monthly_income"].clip(lower=1000)
    df["debt_to_income_ratio"] = (df["monthly_debt"] / df["monthly_income"]).round(4)
    df["application_date"] = df["application_date"].dt.strftime("%Y-%m-%d")

    cols = ["applicant_id", "application_date", "age", "annual_income", "monthly_income", "employment_years",
            "loan_amount", "loan_term_months", "credit_history_years", "bureau_score", "existing_loans",
            "monthly_debt", "debt_to_income_ratio", "number_of_dependents", "employment_type", "education",
            "location", "gender", "marital_status",
            "interest_rate_offered", "loan_approved", "days_past_due_after_disbursal", "collections_status",
            "post_disbursal_payment_ratio", "credit_officer_override", "default_12m"]
    return df[cols]


# ---- columns of a production batch: the API request fields + id/date + target + audit-only columns ----
REQUEST_FIELDS = ["annual_income", "monthly_income", "employment_years", "loan_amount", "loan_term_months",
                  "credit_history_years", "bureau_score", "existing_loans", "monthly_debt",
                  "number_of_dependents", "employment_type", "education", "location"]
AUDIT_ONLY = ["age", "gender"]       # never sent to the scoring API; used by the fairness audit only
TARGET = "default_12m"


def production_batch(n: int, seed: int, scenario: dict, start: str, end: str, id_prefix: str) -> pd.DataFrame:
    """A 'production' batch = what the live system sees: request fields + target (arrives 12 months later)."""
    df = generate(n, seed=seed, scenario=scenario, start=start, end=end, reapply_rate=0.0, id_prefix=id_prefix)
    return df[["applicant_id", "application_date"] + REQUEST_FIELDS + AUDIT_ONLY + [TARGET]]


def main() -> None:
    raw = generate()
    raw.to_csv(DATA_RAW / "applications.csv", index=False)
    production_batch(3000, 7, NORMAL_PROD, "2024-04-01", "2024-09-30", "N").to_csv(DATA_PRODUCTION / "batch_normal.csv", index=False)
    production_batch(3000, 8, DOWNTURN, "2024-04-01", "2024-09-30", "D").to_csv(DATA_PRODUCTION / "batch_drifted.csv", index=False)
    print(f"raw applications : {raw.shape}  default rate = {raw[TARGET].mean():.3f}")
    print(f"written          : {DATA_RAW / 'applications.csv'}")
    print(f"production       : batch_normal.csv, batch_drifted.csv (3,000 rows each, with TRUE target for later monitoring)")


if __name__ == "__main__":
    main()
