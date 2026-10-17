"""Simple fairness audit with pandas only (formal libraries such as Fairlearn / AIF360 are an OPTIONAL extension).

IMPORTANT: gender and age are used ONLY here, to MEASURE outcomes.  They are never model inputs.

Definitions (favourable outcome = the applicant is NOT declined, i.e. APPROVE or REFER_TO_UNDERWRITER)
  approval rate             share of a group that is not declined
  adverse impact ratio AIR  lowest group approval rate / highest group approval rate.
                            Four-fifths rule of thumb: AIR < 0.80 is a red flag that needs investigation.
                            Example: women 0.72 approved, men 0.80 approved -> AIR = 0.72/0.80 = 0.90 (passes).
  equal-opportunity gap     difference in the share of GOOD customers (those who really repaid) that were approved.
                            "Among people who would have repaid, are both groups approved equally often?"
                            Example: 0.88 for men vs 0.84 for women -> gap = 0.04.
  default rate              observed default rate of the group (differences can be legitimate - e.g. different incomes -
                            but must be explained, never assumed).
A passing number is evidence, not a certificate: AIR ignores WHY groups differ, and thresholds are a policy / legal matter
(rules differ by country).

Run:  python -m src.governance.fairness_audit
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import project as P
from src.common.artifact import load_artifact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, REPORTS
from src.models.metrics import credit_metrics
from src.models.pipeline import fit_score, load_splits

AGE_BANDS = [(0, 25, "<25"), (25, 35, "25-34"), (35, 50, "35-49"), (50, 200, "50+")]


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [a for a, _, _ in AGE_BANDS] + [AGE_BANDS[-1][1]], labels=[l for _, _, l in AGE_BANDS], right=False).astype(str)


def group_table(groups: pd.Series, y: pd.Series, declined: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame(dict(g=groups.to_numpy(), y=y.to_numpy(), declined=declined))
    rows = []
    for g, part in d.groupby("g", sort=True):
        good = part[part.y == 0]
        rows.append(dict(group=g, n=len(part), approval_rate=1 - part.declined.mean(), default_rate=part.y.mean(),
                         approved_among_repaid=1 - good.declined.mean() if len(good) else np.nan))
    return pd.DataFrame(rows).round(4)


def adverse_impact_ratio(approval_rates) -> float:
    r = np.asarray(approval_rates, dtype=float)
    return float(r.min() / r.max())


def equal_opportunity_gap(rates) -> float:
    r = np.asarray(rates, dtype=float)
    return float(r.max() - r.min())


def summarize(t: pd.DataFrame) -> dict:
    return dict(air=round(adverse_impact_ratio(t.approval_rate), 4), eo_gap=round(equal_opportunity_gap(t.approved_among_repaid), 4),
                lowest_group=str(t.loc[t.approval_rate.idxmin(), "group"]), highest_group=str(t.loc[t.approval_rate.idxmax(), "group"]))


def cutoff_for_decline_rate(p: np.ndarray, decline_rate: float) -> float:
    return float(np.quantile(p, 1 - decline_rate))


def audit_scores(df: pd.DataFrame, p: np.ndarray, cutoff: float, auto_cutoff: float | None = None) -> dict:
    """cutoff = decline threshold (PD >= cutoff is declined).  auto_cutoff = refer threshold: PD < auto_cutoff is auto-approved (LOW band)."""
    declined = p >= cutoff
    g = group_table(df["gender"], df[P.TARGET], declined)
    a = group_table(age_band(df["age"]), df[P.TARGET], declined)
    res = dict(gender=g, gender_summary=summarize(g), age=a, age_summary=summarize(a), decline_rate=float(declined.mean()))
    d = pd.DataFrame(dict(g=df["gender"].to_numpy(), declined=declined))
    dr = d.groupby("g").declined.mean()
    res["decline_rate_ratio"] = round(float(dr.max() / dr.min()), 3)          # higher denial rate / lower denial rate
    if auto_cutoff is not None:
        auto = pd.DataFrame(dict(g=df["gender"].to_numpy(), auto=p < auto_cutoff)).groupby("g").auto.mean()
        res["auto_approval_by_gender"] = auto.round(4).to_dict()
        res["auto_approval_air"] = round(float(auto.min() / auto.max()), 4)    # stricter view: only instant approvals count as favourable
    return res


def variant_table(train, valid, test, base_feats: list[str], decline_rate: float, auto_rate: float, C: float = 0.5) -> pd.DataFrame:
    """Train several feature sets (EXPERIMENT ONLY - never saved) and compare accuracy and fairness at the SAME decline rate."""
    variants = [
        ("V0 + fine location + age", base_feats + ["location", "age"]),
        ("V1 + fine location (no age)", base_feats + ["location"]),
        ("V2 FINAL: no age, no location at all", base_feats),
        ("V2b + coarse city_tier only", base_feats + ["city_tier"]),
        ("V3 + age (no location)", base_feats + ["age"]),
        ("V4 + gender (NOT ALLOWED, illustration)", base_feats + ["gender"]),
    ]
    rows = []
    for name, feats in variants:
        pipe, _, _ = fit_score(feats, "logreg", C, train, valid, test)
        p = pipe.predict_proba(test[feats])[:, 1]
        m = credit_metrics(test[P.TARGET], p)
        r = audit_scores(test, p, cutoff_for_decline_rate(p, decline_rate), cutoff_for_decline_rate(p, 1 - auto_rate))
        rows.append(dict(variant=name, n_features=len(feats), test_auc=m["roc_auc"], test_ks=m["ks_stat"],
                         gender_AIR=r["gender_summary"]["air"], gender_AIR_auto=r["auto_approval_air"], gender_decline_ratio=r["decline_rate_ratio"],
                         gender_EO_gap=r["gender_summary"]["eo_gap"], age_AIR=r["age_summary"]["air"]))
    return pd.DataFrame(rows)


def run_audit(art: dict | None = None) -> dict:
    prm = load_params()
    art = art or load_artifact(CHAMPION_PATH)
    train, valid, test = load_splits()
    feats = art["input_columns"]
    p = art["pipeline"].predict_proba(test[feats])[:, 1]
    cutoff = art["threshold"]
    refer = art["extra"].get("cutoffs", {}).get("refer", P.REFER_CUTOFF)
    res = audit_scores(test, p, cutoff, refer)
    res["decline_cutoff"] = cutoff
    res["auto_rate"] = float((p < refer).mean())
    res["min_air"] = prm["fairness"]["min_adverse_impact_ratio"]
    res["gate_attribute"] = prm["fairness"]["gate_attribute"]
    res["pass"] = bool(res[f"{res['gate_attribute']}_summary"]["air"] >= res["min_air"])
    return dict(res, p=p, test=test, train=train, valid=valid, feats=feats)


def main() -> None:
    res = run_audit()
    print(f"FAIRNESS AUDIT of champion on the TEST set ({len(res['test'])} applicants); decision cutoff: decline if PD >= {res['decline_cutoff']}")
    print(f"overall decline rate {res['decline_rate']:.1%}\n")
    print("1. BY GENDER (gender was NEVER a model input)")
    print(res["gender"].to_string(index=False))
    s = res["gender_summary"]
    print(f"   adverse impact ratio = {s['lowest_group']} / {s['highest_group']} approval = {s['air']}  "
          f"({'PASS' if s['air'] >= res['min_air'] else 'FAIL'} vs four-fifths rule {res['min_air']});  equal-opportunity gap = {s['eo_gap']}")
    print(f"   denial-rate ratio (higher/lower) = {res['decline_rate_ratio']};  stricter view - share AUTO-approved (LOW band): "
          f"{res['auto_approval_by_gender']} -> AIR {res['auto_approval_air']}")
    print("\n2. BY AGE BAND (age was NEVER a model input; shown for monitoring)")
    print(res["age"].to_string(index=False))
    s2 = res["age_summary"]
    print(f"   adverse impact ratio = {s2['air']}   equal-opportunity gap = {s2['eo_gap']}")
    print("   Reading: younger applicants have thinner credit files and shorter employment, which the model legitimately uses;\n"
          "   whether that is acceptable for age is a POLICY question for Compliance (see docs/REGULATORY_REVIEW_FLOW.md).")
    prm = load_params()
    print("\n3. BEFORE / AFTER REMOVING PROXIES (same decline rate for every variant, trained on the same time-based train split)")
    vt = variant_table(res["train"], res["valid"], res["test"], res["feats"], res["decline_rate"], res["auto_rate"], prm["model"]["C"])
    print(vt.to_string(index=False))
    print("   Reading: fine-grained location (V1) buys almost no accuracy but WIDENS the gap between women and men (auto-approval AIR and\n"
          "   denial ratio get worse): in this data some cities have more women applicants AND a local economic effect, so `location` acts as a\n"
          "   PROXY for gender.  The coarse city_tier (V2b) carries no such effect here.  Decision recorded in the registry: fine location excluded.\n"
          "   V4 is shown only to illustrate why gender is excluded: the audit would see its effect directly.  Age (V3) is studied in\n"
          "   `python -m src.governance.age_experiment`.")
    REPORTS.mkdir(exist_ok=True)
    out = dict(decline_cutoff=res["decline_cutoff"], min_air=res["min_air"], gate_attribute=res["gate_attribute"], pass_=res["pass"],
               overall_decline_rate=round(res["decline_rate"], 4), denial_rate_ratio=res["decline_rate_ratio"],
               auto_approval_by_gender=res["auto_approval_by_gender"], auto_approval_air=res["auto_approval_air"],
               gender=res["gender"].to_dict("records"), gender_summary=res["gender_summary"],
               age=res["age"].to_dict("records"), age_summary=res["age_summary"], variants=vt.to_dict("records"))
    (REPORTS / "fairness_report.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwritten reports/fairness_report.json   (gate: {res['gate_attribute']} AIR >= {res['min_air']} -> {'PASS' if res['pass'] else 'FAIL'})")


if __name__ == "__main__":
    main()
