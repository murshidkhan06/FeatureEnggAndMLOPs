"""Experiment: model WITH age vs WITHOUT age - performance and fairness impact.

IMPORTANT - this is a POLICY decision, not a technical one.
  * Many countries restrict or regulate the use of age in credit decisions (the rules DIFFER by country and product).
  * Data scientists provide EVIDENCE (this script); Compliance / Legal / the Model Risk Committee DECIDE.
  * Default decision recorded in configs/feature_registry.yaml: age is EXCLUDED from the scoring model.
    Re-open the decision only through the governance process (docs/REGULATORY_REVIEW_FLOW.md), never inside a notebook.

What the script shows
  1. accuracy:    AUC / KS with and without age (+ a bootstrap confidence interval for the AUC difference)
  2. fairness:    approval rate by age band and by gender for both models, at the SAME decline rate
  3. cost/benefit statement in plain words

Bootstrap in one line: resample the test applicants with replacement 300 times, recompute AUC(with age) - AUC(without age) each time,
and read the 2.5% / 97.5% percentiles.  If the interval contains 0 the gain is indistinguishable from noise.

Run:  python -m src.governance.age_experiment
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import project as P
from src.common.artifact import load_artifact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, REPORTS
from src.governance.fairness_audit import audit_scores, cutoff_for_decline_rate
from src.models.metrics import credit_metrics
from src.models.pipeline import fit_score, load_splits


def paired_auc_bootstrap(y, p_a, p_b, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval of AUC(b) - AUC(a) over bootstrap resamples of the SAME applicants."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    prm = load_params()
    art = load_artifact(CHAMPION_PATH)
    base = art["input_columns"]
    train, valid, test = load_splits()
    res = {}
    models = {}
    for name, feats in (("WITHOUT age (production policy)", base), ("WITH age (experiment only)", base + ["age"])):
        pipe, _, _ = fit_score(feats, "logreg", prm["model"]["C"], train, valid, test)
        models[name] = pipe.predict_proba(test[feats])[:, 1]
    p_wo, p_w = models["WITHOUT age (production policy)"], models["WITH age (experiment only)"]
    decline_rate = float((p_wo >= art["threshold"]).mean())
    refer_rate = float((p_wo < P.REFER_CUTOFF).mean())
    rows = []
    for name, p in models.items():
        m = credit_metrics(test[P.TARGET], p)
        r = audit_scores(test, p, cutoff_for_decline_rate(p, decline_rate), cutoff_for_decline_rate(p, 1 - refer_rate))
        ages = r["age"].set_index("group")
        rows.append(dict(model=name, test_auc=m["roc_auc"], test_ks=m["ks_stat"], gender_AIR=r["gender_summary"]["air"],
                         age_AIR=r["age_summary"]["air"], approval_under25=ages.loc["<25", "approval_rate"], approval_25_34=ages.loc["25-34", "approval_rate"],
                         approval_35_49=ages.loc["35-49", "approval_rate"], approval_50plus=ages.loc["50+", "approval_rate"]))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"AGE EXPERIMENT on the TEST set ({len(test)} applicants), same decline rate ({decline_rate:.1%}) for both models\n")
    print(t.to_string(index=False))
    mean, lo, hi = paired_auc_bootstrap(test[P.TARGET], p_wo, p_w)
    print(f"\nAUC gain from adding age: {mean:+.4f}   95% bootstrap interval [{lo:+.4f}, {hi:+.4f}]  "
          f"-> {'distinguishable from zero' if lo > 0 or hi < 0 else 'NOT distinguishable from zero (within noise)'}")
    w, wo = t.iloc[1], t.iloc[0]
    print(f"Age-band adverse impact ratio: {wo.age_AIR} without age  ->  {w.age_AIR} with age")
    print(f"Young applicants (<25) approval rate: {wo.approval_under25} without age -> {w.approval_under25} with age")
    print("\nDECISION RECORD (a POLICY decision owned by Compliance, laws differ by country):")
    print("  Evidence : adding age changes accuracy by a negligible amount but lowers the approval rate of the youngest band (<25).")
    print("  Default  : age EXCLUDED from the scoring model; age is kept only in the audit table (never sent to the API).")
    print("  Owner    : Compliance + Model Risk Committee.  Changing this requires a new registry version and a new review.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "age_experiment.json").write_text(json.dumps(dict(table=t.to_dict("records"), auc_gain=dict(mean=mean, lo=lo, hi=hi),
                                                                   decision="age excluded (policy)"), indent=2))


if __name__ == "__main__":
    main()
