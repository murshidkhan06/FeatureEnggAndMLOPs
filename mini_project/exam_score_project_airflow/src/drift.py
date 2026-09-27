"""
src/drift.py

DATA DRIFT detection (PHASE 17): compares the distribution of features in the REFERENCE
dataset (data/reference/training_reference.csv -- a frozen snapshot of the raw training
inputs, written by src/train.py) against a CURRENT dataset (by default, the raw inputs
logged from real predictions in monitoring/predictions.csv).

This module covers TWO kinds of data drift, because they need different statistical tests:

1. NUMERIC drift (`run_ks_drift_check` / `check_drift`) -- the two-sample
   Kolmogorov-Smirnov test, one continuous feature at a time.
   WHY the KS test specifically? It's distribution-free (no assumption that a feature is
   normally distributed), works on any continuous numeric feature, and gives both a
   distance statistic (how different the two distributions look) and a p-value (how likely
   that difference is to have occurred by chance if the two samples actually came from the
   same distribution).

2. CATEGORICAL drift (`run_chi_square_drift_check` / `check_categorical_drift`) -- a
   chi-square test of independence, one categorical feature at a time. A numeric test like
   KS doesn't apply to labels like `city` or `income_bracket` (there's no "distance" between
   "Pune" and "Mumbai"); instead we compare how often each CATEGORY occurs in the reference
   vs. current data using frequency counts. See that function's docstring for the full
   mechanics and how this differs from the numeric case.

IMPORTANT CAVEAT this module deliberately keeps front and center for BOTH tests: a low
p-value means the difference is STATISTICALLY significant -- it does NOT automatically mean
the difference is BUSINESS significant. 500 production requests where `attendance_pct`
shifted by half a point can trip p < 0.05 without the model's real-world accuracy moving at
all. Always read a drift report next to the actual effect size (the KS/chi-square statistic
itself, and a look at the two distributions), not the p-value alone.

Neither test here detects CONCEPT drift (a change in the relationship between features and
the target, P(Y|X)) -- both only compare feature distributions, P(X). See
src/concept_drift_demo.py for why that distinction matters and src/performance_monitor.py
for the model-drift/performance-decay check that concept drift actually requires.

Run from the repo root:
    python -m src.drift
    python -m src.drift --current path/to/other_batch.csv --threshold 0.01
    python -m src.drift --categorical-only
"""
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = REPO_ROOT / "data" / "reference" / "training_reference.csv"
PREDICTIONS_LOG = REPO_ROOT / "monitoring" / "predictions.csv"
REPORT_PATH = REPO_ROOT / "reports" / "drift_report.csv"
CATEGORICAL_REPORT_PATH = REPO_ROOT / "reports" / "categorical_drift_report.csv"

NUMERIC_DRIFT_COLUMNS = [
    "study_hours", "attendance_pct", "mock_test_1", "mock_test_2", "mock_test_3",
    "shoe_size", "lucky_number",
]


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def run_ks_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    p_value_threshold: float,
) -> pd.DataFrame:
    """Runs a two-sample KS test per column. Returns a report DataFrame with one row per
    feature: the KS statistic, the p-value, and a boolean `drift` flag."""
    rows = []
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_values = reference_df[col].dropna()
        cur_values = current_df[col].dropna()
        if len(ref_values) < 2 or len(cur_values) < 2:
            continue  # not enough data to test meaningfully

        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        rows.append({
            "feature": col,
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift": bool(p_value < p_value_threshold),
            "reference_n": len(ref_values),
            "current_n": len(cur_values),
        })
    return pd.DataFrame(rows)


def check_drift(
    reference_path: Path = REFERENCE_PATH,
    current_path: Optional[Path] = None,
    p_value_threshold: Optional[float] = None,
) -> pd.DataFrame:
    params = load_params()
    p_value_threshold = p_value_threshold or params["drift"]["ks_p_value_threshold"]

    if not reference_path.exists():
        raise FileNotFoundError(
            f"No reference distribution at {reference_path}. Run `python -m src.train` "
            f"first -- it writes this file automatically."
        )
    reference_df = pd.read_csv(reference_path)

    current_path = current_path or PREDICTIONS_LOG
    if not Path(current_path).exists():
        raise FileNotFoundError(
            f"No current-traffic data at {current_path}. Send some /predict requests "
            f"first (or pass --current pointing at a CSV of raw feature values)."
        )
    current_df = pd.read_csv(current_path)

    report = run_ks_drift_check(reference_df, current_df, NUMERIC_DRIFT_COLUMNS, p_value_threshold)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


def run_chi_square_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    p_value_threshold: float,
) -> pd.DataFrame:
    """Runs a chi-square test of independence per categorical column.

    HOW this differs from the numeric KS test: a numeric feature has an ORDER and a
    DISTANCE (study_hours=2 is closer to study_hours=3 than to study_hours=20), so KS can
    compare the two empirical distribution functions directly. A categorical feature like
    `city` has neither -- "Pune" isn't numerically closer to "Mumbai" than to "Delhi". So
    instead we build a 2xK contingency table (2 = {reference, current}, K = number of
    distinct categories) of RAW COUNTS -- how many times each category appears in each
    dataset -- and ask: "if category frequency didn't depend on which dataset a row came
    from, how likely is it we'd see counts this skewed by chance?" That's exactly what
    `scipy.stats.chi2_contingency` answers.

    Null hypothesis (H0): the category proportions are the SAME in both datasets (category
    is independent of which dataset the row came from -- i.e. no drift).
    Alternative hypothesis (H1): the category proportions DIFFER between datasets.
    A p-value below the threshold means H0 is rejected: proportions differ enough that
    "just sampling noise" is an unlikely explanation.

    A category present in current data but never seen in the reference data (e.g. a brand
    new city) is itself a meaningful signal and is included in the table with a reference
    count of 0, rather than silently dropped.
    """
    rows = []
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_values = reference_df[col].dropna()
        cur_values = current_df[col].dropna()
        if len(ref_values) < 2 or len(cur_values) < 2:
            continue

        categories = sorted(set(ref_values.unique()) | set(cur_values.unique()))
        ref_counts = ref_values.value_counts().reindex(categories, fill_value=0)
        cur_counts = cur_values.value_counts().reindex(categories, fill_value=0)
        contingency = pd.DataFrame({"reference": ref_counts, "current": cur_counts}).T

        # chi2_contingency needs every column (category) to have a nonzero total count
        # across the two rows -- drop any category that's all-zero after the reindex
        # (can't happen given how `categories` was built, but guards against edge cases).
        contingency = contingency.loc[:, contingency.sum(axis=0) > 0]
        if contingency.shape[1] < 2:
            continue  # only one distinct category seen anywhere -- nothing to compare

        statistic, p_value, dof, _expected = stats.chi2_contingency(contingency.values)
        new_categories = sorted(set(cur_values.unique()) - set(ref_values.unique()))
        rows.append({
            "feature": col,
            "chi2_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift": bool(p_value < p_value_threshold),
            "n_categories": contingency.shape[1],
            "new_categories": ", ".join(new_categories) if new_categories else "",
            "reference_n": len(ref_values),
            "current_n": len(cur_values),
        })
    return pd.DataFrame(rows)


def check_categorical_drift(
    reference_path: Path = REFERENCE_PATH,
    current_path: Optional[Path] = None,
    columns: Optional[list[str]] = None,
    p_value_threshold: Optional[float] = None,
) -> pd.DataFrame:
    params = load_params()
    cat_cfg = params["drift"]
    p_value_threshold = p_value_threshold or cat_cfg["categorical_p_value_threshold"]
    columns = columns or cat_cfg["categorical_columns"]

    if not reference_path.exists():
        raise FileNotFoundError(
            f"No reference distribution at {reference_path}. Run `python -m src.train` "
            f"first -- it writes this file automatically."
        )
    reference_df = pd.read_csv(reference_path)

    current_path = current_path or PREDICTIONS_LOG
    if not Path(current_path).exists():
        raise FileNotFoundError(
            f"No current-traffic data at {current_path}. Send some /predict requests first."
        )
    current_df = pd.read_csv(current_path)

    report = run_chi_square_drift_check(reference_df, current_df, columns, p_value_threshold)
    CATEGORICAL_REPORT_PATH.parent.mkdir(exist_ok=True)
    report.to_csv(CATEGORICAL_REPORT_PATH, index=False)
    return report


def print_categorical_report(report: pd.DataFrame, p_value_threshold: float) -> None:
    if report.empty:
        print("No comparable categorical columns / not enough data to run a drift check yet.")
        return

    print(f"\nCategorical drift report (chi-square test, p_value_threshold={p_value_threshold})")
    print("-" * 78)
    print(f"{'feature':<18}{'chi2':>10}{'p_value':>12}{'drift':>10}   new_categories")
    print("-" * 78)
    for _, row in report.iterrows():
        flag = "YES" if row["drift"] else "NO"
        print(f"{row['feature']:<18}{row['chi2_statistic']:>10.3f}{row['p_value']:>12.3f}"
              f"{flag:>10}   {row['new_categories']}")

    drifted = report[report["drift"]]
    if len(drifted) > 0:
        print(f"\n[ALERT] Categorical drift detected in {len(drifted)} feature(s): "
              f"{', '.join(drifted['feature'])}")
    else:
        print("\nNo statistically significant categorical drift detected.")


def print_report(report: pd.DataFrame, p_value_threshold: float) -> None:
    if report.empty:
        print("No comparable numeric columns / not enough data to run a drift check yet.")
        return

    print(f"\nData drift report (KS test, p_value_threshold={p_value_threshold})")
    print("-" * 68)
    print(f"{'feature':<20}{'statistic':>12}{'p_value':>12}{'drift':>12}")
    print("-" * 68)
    for _, row in report.iterrows():
        flag = "YES" if row["drift"] else "NO"
        print(f"{row['feature']:<20}{row['statistic']:>12.3f}{row['p_value']:>12.3f}{flag:>12}")

    drifted = report[report["drift"]]
    if len(drifted) > 0:
        print(f"\n[ALERT] Drift detected in {len(drifted)} feature(s): "
              f"{', '.join(drifted['feature'])}")
        print("Reminder: statistical significance != business significance. Inspect the "
              "actual distributions before deciding this needs action (see src/retrain.py "
              "for how this project gates retraining decisions).")
    else:
        print("\nNo statistically significant drift detected.")


def main():
    parser = argparse.ArgumentParser(description="Check for data drift against the training reference.")
    parser.add_argument("--current", type=Path, default=None,
                         help="CSV of current/production raw feature values (default: monitoring/predictions.csv)")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Numeric KS-test p-value threshold (default: from params.yaml)")
    parser.add_argument("--categorical-only", action="store_true",
                         help="Only run the categorical (chi-square) drift check")
    parser.add_argument("--numeric-only", action="store_true",
                         help="Only run the numeric (KS test) drift check")
    args = parser.parse_args()

    params = load_params()

    if not args.categorical_only:
        threshold = args.threshold or params["drift"]["ks_p_value_threshold"]
        report = check_drift(current_path=args.current, p_value_threshold=threshold)
        print_report(report, threshold)

    if not args.numeric_only:
        cat_threshold = params["drift"]["categorical_p_value_threshold"]
        cat_report = check_categorical_drift(current_path=args.current, p_value_threshold=cat_threshold)
        print_categorical_report(cat_report, cat_threshold)


if __name__ == "__main__":
    main()
