"""MONTHLY BATCH SCORING - the realistic way an early-warning model runs in a bank.

WHY batch (and not only the API)?  Collections needs ONE ranked call list per month, for ALL customers, produced after
month-end.  Nobody needs a prediction for a single customer at 3 a.m.  The API (POST /predict) exists for ad-hoc lookups
(a relationship manager opens one customer) and uses the SAME model and the SAME derived-feature code.

WHAT THIS SCRIPT DOES (for as_of = a month-end)
  1. read the raw tables and keep ONLY rows available on as_of  (build_snapshot_features filters first: availability rule)
  2. compute the 9 base features, add the 3 derived ones (add_derived_features - the same function as training / the API)
  3. score every eligible customer with models/champion.joblib  -> probability of default in the next 6 months
  4. rank customers, mark the top `top_pct` (10 %) as "contact recommended" and write  reports/monthly_scores.csv

There are NO labels here: on 2025-06-30 nobody knows yet who will default before 2025-12-31.  That is exactly the production situation.

TINY EXAMPLE of the contact list:  1,000 scored customers, top 10 % = 100 calls.  If the 100 highest scores contain 35 of the 50 customers
who really default in the next 6 months, recall@10% = 35 / 50 = 0.70 (we only learn this 6 months later).

Run:  python -m src.models.batch_score [--as-of 2025-06-30] [--top-pct 0.10] [--out reports/monthly_scores.csv]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.config import load_params
from src.common.facts import save_fact
from src.common.paths import CHAMPION_PATH, ROOT
from src.features.build_features import add_derived_features, build_snapshot_features, load_raw


def score_month(as_of, art: dict, raw: dict | None = None, top_pct: float = 0.10) -> pd.DataFrame:
    """Score every eligible customer on `as_of`.  Returns one row per customer, highest risk first.

    Columns: customer_id, features_as_of, probability_of_default_6m, risk_band, rank, contact_recommended, model_version."""
    raw = raw or load_raw()
    as_of = pd.Timestamp(as_of)
    known_defaults = raw["defaults"][raw["defaults"]["default_date"] <= as_of]       # only defaults that already happened are known today
    feats = build_snapshot_features(raw["customers"], raw["transactions"], raw["loan_status"], as_of, known_defaults)
    X = add_derived_features(feats)
    p = art["pipeline"].predict_proba(X[art["input_columns"]])[:, 1]
    out = pd.DataFrame({"customer_id": feats["customer_id"], "features_as_of": as_of.date(),
                        "probability_of_default_6m": p.round(4)})
    out["risk_band"] = [P.risk_level(v) for v in p]
    out = out.sort_values("probability_of_default_6m", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = out.index + 1
    n_contact = max(1, int(round(top_pct * len(out))))
    out["contact_recommended"] = out["rank"] <= n_contact
    out["model_version"] = art["version"]
    return out


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=params["retrain"]["today"], help="month-end to score (default: the last month of data)")
    ap.add_argument("--top-pct", type=float, default=params["model"]["top_pct"])
    ap.add_argument("--out", default="reports/monthly_scores.csv", help="relative to the project root")
    a = ap.parse_args()
    art = load_artifact(CHAMPION_PATH)
    scores = score_month(a.as_of, art, top_pct=a.top_pct)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    n = len(scores)
    print(f"model {art['version']} scored {n:,} customers on {a.as_of}  -> {a.out}")
    print(f"mean PD {scores['probability_of_default_6m'].mean():.4f}   risk bands: " +
          ", ".join(f"{b}={int((scores['risk_band'] == b).sum())}" for b in ("HIGH", "MEDIUM", "LOW")))
    print(f"contact list (top {a.top_pct:.0%}): {int(scores['contact_recommended'].sum())} customers, "
          f"PD range {scores.loc[scores['contact_recommended'], 'probability_of_default_6m'].min():.3f} .. {scores['probability_of_default_6m'].max():.3f}")
    save_fact("batch_score", dict(as_of=a.as_of, customers=n, mean_pd=round(float(scores["probability_of_default_6m"].mean()), 4),
                                  contacts=int(scores["contact_recommended"].sum()), bands={b: int((scores["risk_band"] == b).sum()) for b in ("HIGH", "MEDIUM", "LOW")}))
    print("\ntop 10 customers to call first:")
    print(scores.head(10).to_string(index=False))
    print(f"\ncontrast: the champion's stored threshold ({art['threshold']:.4f}) flags "
          f"{float((scores['probability_of_default_6m'] >= art['threshold']).mean()):.1%} of customers (plan {a.top_pct:.0%}); a stress period inflates this.")


if __name__ == "__main__":
    main()
