"""Metrics and business meaning for a credit-risk early-warning model.

WHY these metrics (not accuracy!)
  Only ~4 % of snapshots default.  A model that says "nobody defaults" is 96 % accurate and useless.
  Collections can only call a limited number of customers, so what matters is the QUALITY OF THE RANKING at the top.

  roc_auc          probability that a random defaulter gets a higher score than a random non-defaulter (0.5 = coin flip)
  pr_auc           area under the precision-recall curve (average precision); the honest metric when positives are rare
  recall_at_10pct  of ALL future defaulters, what share sits in the 10 % of customers with the highest scores?
  lift_at_10pct    precision in the top 10 % divided by the overall default rate (3.0 = three times better than random calls)
  ks               max distance between the score distributions of defaulters and non-defaulters (credit-scoring standard)

TINY WORKED EXAMPLE (10 customers, 2 defaulters, contact the top 10 % = 1 customer)
  scores   0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05      defaulter?  1 0 1 0 0 0 0 0 0 0
  top 1 customer has score 0.9 and defaults  -> recall@10% = 1/2 = 0.50
  precision@10% = 1/1 = 1.0, overall default rate 0.2  -> lift = 1.0 / 0.2 = 5.0

CLI:  python -m src.models.evaluate            (evaluates the CHAMPION on the validation and test periods)
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k_index(score: np.ndarray, pct: float) -> np.ndarray:
    """Indices of the ceil(pct * n) highest scores.  Example: n=10, pct=0.10 -> 1 customer."""
    k = max(1, math.ceil(pct * len(score)))
    return np.argsort(-np.asarray(score), kind="stable")[:k]


def recall_at_top(y, score, pct: float = 0.10) -> float:
    y = np.asarray(y)
    return float(y[top_k_index(score, pct)].sum() / max(y.sum(), 1))


def lift_at_top(y, score, pct: float = 0.10) -> float:
    y = np.asarray(y)
    return float(y[top_k_index(score, pct)].mean() / max(y.mean(), 1e-12))


def ks_score(y, score) -> float:
    """KS of the score: max |CDF(defaulters) - CDF(non-defaulters)|.  0 = no separation, 1 = perfect."""
    y, score = np.asarray(y), np.asarray(score)
    return float(ks_2samp(score[y == 1], score[y == 0]).statistic)


def compute_metrics(y, score, top_pct: float = 0.10) -> dict:
    """The standard metric bundle used everywhere (training, retraining, monitoring)."""
    y = np.asarray(y)
    return {"roc_auc": round(float(roc_auc_score(y, score)), 4), "pr_auc": round(float(average_precision_score(y, score)), 4),
            "recall_at_10pct": round(recall_at_top(y, score, top_pct), 4), "lift_at_10pct": round(lift_at_top(y, score, top_pct), 3),
            "ks": round(ks_score(y, score), 4), "base_rate": round(float(y.mean()), 4), "n": int(len(y)),
            "mean_score": round(float(np.mean(score)), 4)}


def business_summary(y, score, top_pct: float = 0.10) -> dict:
    """Translate metrics into collections-team language."""
    y = np.asarray(y)
    idx = top_k_index(score, top_pct)
    caught, total = int(y[idx].sum()), int(y.sum())
    return {"customers_scored": int(len(y)), "customers_contacted": int(len(idx)), "true_defaulters": total,
            "defaulters_reached": caught, "defaulters_missed": total - caught,
            "share_of_defaulters_reached": round(caught / max(total, 1), 3),
            "calls_per_defaulter_reached": round(len(idx) / max(caught, 1), 1),
            "calls_per_defaulter_if_random": round(len(y) / max(total, 1), 1)}


def print_business_summary(b: dict, top_pct: float = 0.10) -> None:
    print(f"  Of {b['customers_scored']:,} customers scored, collections contacts the top {top_pct:.0%} = {b['customers_contacted']:,} customers.")
    print(f"  {b['true_defaulters']} customers really defaulted in the next 6 months; the contact list reaches {b['defaulters_reached']} of them "
          f"({b['share_of_defaulters_reached']:.0%}), and misses {b['defaulters_missed']}.")
    print(f"  Cost of finding one defaulter: {b['calls_per_defaulter_reached']} calls with the model vs {b['calls_per_defaulter_if_random']} calls by random calling.")


def decile_table(y, score) -> pd.DataFrame:
    """Decile 1 = highest scores.  A good model has a steep drop in bad_rate from decile 1 to decile 10."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score)}).sort_values("s", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * 10 // len(d)) + 1
    t = d.groupby("decile").agg(customers=("y", "size"), defaulters=("y", "sum"), bad_rate=("y", "mean"), avg_score=("s", "mean"))
    t["cum_capture"] = t["defaulters"].cumsum() / max(t["defaulters"].sum(), 1)
    return t.round(4).reset_index()


def evaluate_pipeline(pipe, df: pd.DataFrame, features: list[str], target: str, top_pct: float = 0.10) -> tuple[dict, np.ndarray]:
    score = pipe.predict_proba(df[features])[:, 1]
    return compute_metrics(df[target], score, top_pct), score


if __name__ == "__main__":
    from src.common.artifact import load_artifact
    from src.common.config import load_params
    from src.common.paths import CHAMPION_PATH, REPORTS, SNAPSHOTS_PATH
    from src.features.build_features import add_derived_features
    from src.features.definitions import TARGET

    params = load_params()
    art = load_artifact(CHAMPION_PATH)
    snap = add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
    top = params["model"]["top_pct"]
    report = {"model_version": art["version"], "model_kind": art["extra"].get("kind"), "periods": {}}
    print(f"CHAMPION {art['version']} ({art['extra'].get('kind')})  features={art['input_columns']}\n")
    for split in ("train", "val", "test"):
        d = snap[snap["split"] == split]
        m, score = evaluate_pipeline(art["pipeline"], d, art["input_columns"], TARGET, top)
        report["periods"][split] = m
        print(f"[{split}] {d['as_of_date'].min().date()} .. {d['as_of_date'].max().date()}  n={m['n']:,}  default rate={m['base_rate']:.2%}")
        print("   " + "  ".join(f"{k}={m[k]}" for k in ("roc_auc", "pr_auc", "recall_at_10pct", "lift_at_10pct", "ks", "mean_score")))
        if split == "test":
            b = business_summary(d[TARGET], score, top)
            report["business_test"] = b
            report["decile_test"] = decile_table(d[TARGET], score).to_dict("records")
            print("\nBUSINESS MEANING on the TEST period (never used for any decision):")
            print_business_summary(b, top)
            print("\nDECILE TABLE (decile 1 = riskiest 10 % of customers):")
            print(decile_table(d[TARGET], score).to_string(index=False))
            thr = art["threshold"]
            print(f"\nFixed contact threshold from validation (score >= {thr:.4f}) -> contacts {float((score >= thr).mean()):.1%} of test customers"
                  f" (capacity plan: {top:.0%})")
            report["contact_rate_test_at_val_threshold"] = round(float((score >= thr).mean()), 4)
    print("\nCaution: the TRAIN row is optimistic (the model has seen those customers); judge the model on val/test.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    from src.common.facts import save_fact
    save_fact("evaluation", report)
