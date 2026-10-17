"""DATA PROFILE: look at the data BEFORE modelling (shape, target rate, missing values, correlations, outliers, imbalance, proxies).

WHY: every modelling decision in this project (time split, imputation + missing indicator, tree model, metric choice, leakage
checks) is justified by something visible in this report.  Run it first:   python -m src.data.profile_data

TINY EXAMPLE (imbalance):  4 % defaulters -> the rule "nobody defaults" is 96 % accurate and catches 0 defaulters,
which is why we use ROC-AUC / PR-AUC / recall@top-10 % and never accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.facts import save_fact
from src.common.paths import SNAPSHOTS_PATH
from src.features.build_features import add_derived_features, load_raw
from src.features.definitions import BASE_FEATURES, MODEL_FEATURES, TARGET


def main() -> None:
    pd.set_option("display.width", 200)
    raw = load_raw()
    snap = add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
    cu, tx, ls, de = raw["customers"], raw["transactions"], raw["loan_status"], raw["defaults"]
    facts: dict = {}

    print("1) RAW TABLES (what the core-banking systems deliver)")
    facts["raw_tables"] = {}
    for name, df, col in (("customers", cu, "signup_date"), ("transactions", tx, "txn_date"), ("loan_status", ls, "month"), ("defaults", de, "default_date")):
        print(f"   {name:<13} rows={len(df):>8,}  cols={df.shape[1]}  {col}: {df[col].min().date()} .. {df[col].max().date()}")
        facts["raw_tables"][name] = dict(rows=len(df), cols=df.shape[1], first=str(df[col].min().date()), last=str(df[col].max().date()))
    print("\n   transactions by type:")
    t = tx.groupby("txn_type")["amount"].agg(rows="size", mean="mean", median="median", max="max").round(0)
    print(t.to_string())
    n_def = int(de["default_date"].notna().sum())
    print(f"\n   customers that default at some point in the 54 months: {n_def} of {len(de)} ({n_def / len(de):.1%})  "
          "(high on purpose: a bank portfolio would show ~2-4 % per year; we need enough positives to teach with)")
    yr = de["default_date"].dropna().dt.year.value_counts().sort_index()
    print("   defaults per calendar year:", yr.to_dict(), " <- the rise from 2023-09 is the simulated macro stress")
    facts["customers_defaulting"] = n_def
    facts["defaults_per_year"] = {int(k): int(v) for k, v in yr.items()}

    print("\n2) SNAPSHOT DATASET (one row per customer x month-end; target = default in the next 6 months)")
    sp = snap.groupby("split").agg(rows=(TARGET, "size"), customers=("customer_id", "nunique"), positives=(TARGET, "sum"), rate=(TARGET, "mean"),
                                   first=("as_of_date", "min"), last=("as_of_date", "max"))
    sp["rate"] = sp["rate"].round(4)
    sp = sp.loc[["train", "purge", "val", "test", "production"]]
    print(f"   shape {snap.shape}   target = {TARGET}")
    print(sp.to_string())
    rate = float(snap[snap['split'] == 'train'][TARGET].mean())
    print(f"\n   IMBALANCE: {rate:.1%} positives in the training period.  'Nobody defaults' would be {1 - rate:.1%} accurate and useless.")
    facts["shape"] = list(snap.shape)
    facts["split_table"] = {k: dict(rows=int(r.rows), customers=int(r.customers), positives=int(r.positives), rate=float(r.rate), first=str(r["first"].date()), last=str(r["last"].date())) for k, r in sp.iterrows()}
    facts["target_rate_train"] = round(rate, 4)

    tr = snap[snap["split"] == "train"]
    print("\n3) MISSING VALUES (training rows)")
    miss = (tr[MODEL_FEATURES].isna().mean() * 100).round(2)
    print(miss[miss > 0].to_string() if (miss > 0).any() else "   none")
    nc = tr["credit_utilization"].isna().mean()
    print(f"   credit_utilization: missing for customers WITHOUT a credit card ({nc:.1%}) - MAR: depends on segment (affluent 2 % no card, mass 22 %)")
    print("   income_stability: undefined with < 2 salary credits in 6 months - MNAR: the missingness itself is a risk signal (job loss)")
    by = tr.assign(inc_missing=tr["income_stability"].isna()).groupby("inc_missing")[TARGET].agg(["size", "mean"]).round(4)
    print(by.to_string())
    n_miss = int(tr["income_stability"].isna().sum())
    print(f"   -> only {n_miss} training rows have it missing (rare in calm months), but every one of those customers has stopped receiving a salary; the pipeline")
    print("      still keeps a MISSING INDICATOR (SimpleImputer add_indicator=True) because under stress this becomes more common (see drifted batch).")
    facts["missing_rates_pct"] = {k: float(v) for k, v in miss.items()}
    facts["income_stability_missing_rows_train"] = int(tr["income_stability"].isna().sum())
    facts["income_stability_missing_default_rate"] = {str(k): dict(rows=int(v["size"]), rate=float(v["mean"])) for k, v in by.iterrows()}

    print("\n4) CORRELATION WITH THE TARGET (training rows, Pearson with the 0/1 target; a quick screen, not a verdict)")
    c = tr[MODEL_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3)
    print(c.to_string())
    facts["top_correlations_with_target"] = c.head(8).to_dict()
    print("\n   strongest FEATURE-FEATURE correlations (multicollinearity):")
    cm = tr[BASE_FEATURES].corr().abs()
    pairs = [(a, b, round(float(cm.loc[a, b]), 3)) for i, a in enumerate(BASE_FEATURES) for b in BASE_FEATURES[i + 1:]]
    pairs = sorted(pairs, key=lambda x: -x[2])[:5]
    for a, b, v in pairs:
        print(f"   {a:<26}{b:<26}|r| = {v}")
    facts["top_feature_pairs"] = [dict(a=a, b=b, abs_r=v) for a, b, v in pairs]

    print("\n5) OUTLIERS (one-off huge credits: property sale / inheritance)")
    oc = tx[tx["txn_type"] == "other_credit"]
    typical = float(oc["amount"].median())
    big = oc[oc["amount"] > 8 * typical]
    print(f"   median other_credit {typical:,.0f};  credits above 8x the median: {len(big)} on {big['customer_id'].nunique()} customers; largest {oc['amount'].max():,.0f}")
    q = snap["average_balance"].quantile([0.5, 0.99, 0.999]).round(0)
    print(f"   average_balance quantiles: median {q.iloc[0]:,.0f}, p99 {q.iloc[1]:,.0f}, p99.9 {q.iloc[2]:,.0f}, max {snap['average_balance'].max():,.0f}  -> long right tail: log_average_balance tames it, trees are robust")
    facts["outlier_credits"] = dict(count=int(len(big)), customers=int(big["customer_id"].nunique()), max=float(oc["amount"].max()))

    print("\n6) SENSITIVE / PROXY ATTRIBUTE: segment (NOT used as a model input; see registry: Rejected)")
    seg = snap.merge(cu[["customer_id", "segment"]], on="customer_id").query("split == 'train'")
    st = seg.groupby("segment").agg(rows=(TARGET, "size"), default_rate=(TARGET, "mean"), income_stability=("income_stability", "mean"),
                                    missing_income_stability=("income_stability", lambda s: s.isna().mean())).round(4)
    print(st.to_string())
    print("   small_business customers have irregular income by construction: income_stability and salary features act as PROXIES for the segment.")
    facts["segment_table"] = st.reset_index().to_dict("records")

    print("\n7) LABEL LAG: the label of as_of is known only 6 months later")
    print(f"   last as_of with a known label (data ends 2025-06-30): 2024-12-31.  The newest 6 monthly snapshots have NO label yet - they are what we score in production.")
    save_fact("data_profile", facts)


if __name__ == "__main__":
    main()
