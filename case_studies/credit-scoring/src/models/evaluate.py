"""Evaluate the CHAMPION on the untouched TEST set: metrics, stability, calibration, bands and business meaning.

Run:  python -m src.models.evaluate

What each block tells you
  1. Metrics     ROC-AUC / Gini / KS / PR-AUC / recall at 80% approval (formulas: src/models/metrics.py)
  2. Stability   KS test between the SCORE distribution on train and on test (same two-sample KS used for drift).
                 A big D here means the model scores new customers very differently from the ones it learned from.
  3. Calibration "when the model says 15%, do about 15% default?"  (reliability table, deciles)
  4. Bands       share of applicants and observed default rate in LOW / MEDIUM / HIGH
  5. Business    money view with STATED assumptions (loss given default, margin) at a fixed approval rate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import project as P
from src.common.artifact import load_artifact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, REPORTS
from src.models.metrics import business_view, calibration_table, credit_metrics
from src.models.pipeline import load_splits


def band_table(y, p) -> pd.DataFrame:
    d = pd.DataFrame(dict(y=np.asarray(y), p=np.asarray(p)))
    d["band"] = [P.risk_level(v) for v in d.p]
    t = d.groupby("band").agg(applicants=("y", "size"), mean_pd=("p", "mean"), observed_default_rate=("y", "mean")).reindex(["LOW", "MEDIUM", "HIGH"])
    t["share_of_applicants"] = t.applicants / len(d)
    t["decision"] = [P.DECISION_HINT[b] for b in t.index]
    return t.round(4).reset_index()


def evaluate(art: dict | None = None) -> dict:
    prm = load_params()
    art = art or load_artifact(CHAMPION_PATH)
    pipe, feats = art["pipeline"], art["input_columns"]
    train, valid, test = load_splits()
    scores = {n: pipe.predict_proba(d[feats])[:, 1] for n, d in (("train", train), ("valid", valid), ("test", test))}
    res = dict(version=art["version"], features=feats)
    res["metrics"] = {n: credit_metrics(d[P.TARGET], scores[n]) for n, d in (("train", train), ("valid", valid), ("test", test))}
    ks = ks_2samp(scores["train"], scores["test"])
    res["score_stability_ks_train_vs_test"] = dict(D=round(float(ks.statistic), 4), p_value=float(ks.pvalue))
    res["calibration_test"] = calibration_table(test[P.TARGET], scores["test"]).to_dict("records")
    res["bands_test"] = band_table(test[P.TARGET], scores["test"]).to_dict("records")
    b = prm["business"]
    res["business_test"] = business_view(test["loan_amount"], test[P.TARGET], scores["test"], b["approval_rate"],
                                         b["loss_given_default"], b["margin_on_good_loan"])
    res["comparison_test"] = dict(baseline_logreg_raw=art["extra"].get("baseline_test"), champion=res["metrics"]["test"],
                                  benchmark_hgb=art["extra"].get("benchmark_hgb_test"))
    return res


def main() -> None:
    res = evaluate()
    print(f"CHAMPION {res['version']}  features ({len(res['features'])}): {res['features']}\n")
    print("1. METRICS (train / valid / test; test is the honest report card)")
    print(pd.DataFrame(res["metrics"]).T.to_string())
    m = res["metrics"]["test"]
    print(f"\n   Reading: AUC {m['roc_auc']} -> Gini {m['gini']} (=2*AUC-1).  KS {m['ks_stat']}: the biggest gap between the score CDFs of defaulters and "
          f"good customers.\n   If we decline the riskiest 20% we catch {m['recall_at_80_approval']:.0%} of all defaulters.")
    s = res["score_stability_ks_train_vs_test"]
    print(f"\n2. STABILITY  KS test of train scores vs test scores: D={s['D']}, p={s['p_value']:.3g}  "
          f"({'small shift: stable' if s['D'] < 0.10 else 'LARGE shift: investigate'})")
    print("\n3. CALIBRATION on test (deciles of predicted PD)")
    print(pd.DataFrame(res["calibration_test"]).to_string(index=False))
    print("\n4. RISK BANDS on test")
    print(pd.DataFrame(res["bands_test"]).to_string(index=False))
    bz = res["business_test"]
    print(f"\n5. BUSINESS MEANING on test (approve safest {bz['approval_rate']:.0%}, decline the rest; LGD and margin from params.yaml)")
    print(f"   expected credit loss if we approved everybody : INR {bz['expected_loss_without_model']:,}")
    print(f"   declined applicants                           : {bz['declined']}  (defaulters {bz['defaulters_declined']}, good customers {bz['good_customers_declined']})")
    print(f"   loss avoided                                  : INR {bz['loss_avoided']:,}")
    print(f"   margin lost on good customers we declined     : INR {bz['margin_lost']:,}")
    print(f"   NET BENEFIT of using the model                : INR {bz['net_benefit']:,}")
    print("\n6. MODEL COMPARISON on test")
    print(pd.DataFrame(res["comparison_test"]).T.to_string())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
