"""THE LEAKAGE LESSON: how a feature that "looks too good" silently destroys a credit model - and how to fix it.

Run:  python -m src.features.leakage_demo

LEAKAGE = the model sees information at TRAINING time that will not exist at PREDICTION time.
The one question to ask of every feature:  "Would this number genuinely be available on as_of_date?"

THE STORY IN FOUR ACTS
  Act 1  average_balance_leaky   An analyst builds the training table when the outcome is confirmed (as_of + 7 months) and
                                 re-uses the query "average balance over the last 90 days" with THAT date.  Customers who default get their account
                                 swept (balance collapses to ~0) AFTER the default -> the feature is a perfect-looking
                                 default detector.  AUC explodes.  At scoring time the future balance does not exist,
                                 only the normal average_balance does -> live AUC collapses.
  Act 2  second, subtler leak    dpd_target_period, recoveries_target_period, collections_status_latest: columns that only
                                 exist BECAUSE the customer already defaulted (default = 90+ days past due, recoveries and
                                 collections come after).  Added to the model they give AUC ~1.0.
  Act 3  the FIX                 prediction timestamp + availability rule (available_at <= as_of_date) + leakage-safe
                                 feature pipeline + future-perturbation test + registry gate.
  Act 4  split leakage           a RANDOM split of snapshots leaks too: the same customer appears in train and test.

TINY EXAMPLE (why Act 1 works)  customer defaults 2022-10-31, as_of = 2022-06-30, extract date 2023-01-31:
   month-end balances Apr-Jun 2022: 90k, 85k, 80k  -> average_balance (legit, up to as_of) = 85k
   balances Nov-Jan (swept in Oct): 6k, 4k, 3k      -> leaky (90 days up to 2023-01-31) = 4.3k   <- screams "defaulter"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.config import load_params
from src.common.facts import save_fact
from src.features.build_features import (add_derived_features, build_snapshot_features, load_raw, month_end_shift)
from src.features.definitions import BALANCE_DAYS, HORIZON_MONTHS, MODEL_FEATURES, TARGET
from src.features.leakage_checks import future_perturbation_test
from src.models.pipelines import make_hgb


# ----------------------------------------------------------------------------------------------------------
# the WRONG feature builders (each peeks beyond as_of_date).  Same signature as the safe builder.
# ----------------------------------------------------------------------------------------------------------
LEAK_EXTRACT_MONTHS = HORIZON_MONTHS + 1     # the training table is assembled once the 6-month outcome is CONFIRMED (+1 month)


def average_balance_leaky(customers, transactions, loan_status, as_of) -> pd.DataFrame:
    """WRONG: mean balance over the 90 days ending at the data-EXTRACT date = as_of + 7 months (outcome window + 1 month).

    Same code as the legitimate feature, only the date is wrong.  It includes the post-default balance collapse."""
    end = month_end_shift(as_of, LEAK_EXTRACT_MONTHS)
    t = transactions[(transactions["txn_date"] > end - pd.Timedelta(days=BALANCE_DAYS)) & (transactions["txn_date"] <= end)]
    return t.groupby("customer_id")["balance_after"].mean().rename("average_balance_leaky").reset_index()


def average_balance_all_history(transactions, as_of=None, extract_date=None) -> pd.Series:
    """WRONG (variant): mean balance over ALL history up to the data-extract date (the end of the data)."""
    t = transactions if extract_date is None else transactions[transactions["txn_date"] <= extract_date]
    return t.groupby("customer_id")["balance_after"].mean().rename("average_balance_all_history")


def dpd_target_period(customers, transactions, loan_status, as_of) -> pd.DataFrame:
    """WRONG: maximum days_past_due INSIDE the 6-month target window (ignores the publication lag, uses the future)."""
    end = month_end_shift(as_of, HORIZON_MONTHS)
    w = loan_status[(loan_status["month"] > as_of) & (loan_status["month"] <= end)]
    return w.groupby("customer_id")["days_past_due"].max().rename("dpd_target_period").reset_index()


def recoveries_target_period(customers, transactions, loan_status, as_of) -> pd.DataFrame:
    """WRONG: recoveries collected inside the target window.  Recoveries only exist after a default."""
    end = month_end_shift(as_of, HORIZON_MONTHS)
    w = loan_status[(loan_status["month"] > as_of) & (loan_status["month"] <= end)]
    return w.groupby("customer_id")["recovery_amount"].sum().rename("recoveries_target_period").reset_index()


def collections_status_latest(customers, transactions, loan_status, as_of) -> pd.DataFrame:
    """WRONG: collections status in the LATEST extract (after the outcome) as 1/0."""
    last = loan_status.sort_values("month").groupby("customer_id").tail(1)
    return pd.DataFrame({"customer_id": last["customer_id"].to_numpy(),
                         "collections_status_latest": (last["collections_status"] == "in_collections").astype(int).to_numpy()})


LEAKY_BUILDERS = {"average_balance_leaky": average_balance_leaky, "dpd_target_period": dpd_target_period,
                  "recoveries_target_period": recoveries_target_period, "collections_status_latest": collections_status_latest}


def add_leaky_columns(snap: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """Attach the four leaky columns to snapshot rows (customers with no data in the window get a neutral fill)."""
    parts = []
    for as_of, g in snap.groupby("as_of_date"):
        g = g.copy()
        for name, fn in LEAKY_BUILDERS.items():
            g = g.merge(fn(raw["customers"], raw["transactions"], raw["loan_status"], pd.Timestamp(as_of)), on="customer_id", how="left")
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["average_balance_leaky"] = out["average_balance_leaky"].fillna(0.0)
    for c in ("dpd_target_period", "recoveries_target_period"):
        out[c] = out[c].fillna(0.0)
    out["collections_status_latest"] = out["collections_status_latest"].fillna(0)
    return out


def _auc(model, df, feats):
    return roc_auc_score(df[TARGET], model.predict_proba(df[feats])[:, 1])


def _fit(df, feats):
    return make_hgb(feats, load_params()["model"]["hgb"]).fit(df[feats], df[TARGET])


def main() -> None:
    params = load_params()
    raw = load_raw()
    snap = add_derived_features(pd.read_csv("data/processed/snapshots.csv", parse_dates=["as_of_date"]))
    sp = params["splits"]
    keep = snap[(snap["as_of_date"] <= pd.Timestamp("2023-08-31"))].copy()           # labelled + before the macro stress
    keep = add_leaky_columns(keep, raw)
    tr, va, te = (keep[keep["split"] == s] for s in ("train", "val", "test"))
    live = keep[(keep["as_of_date"] >= pd.Timestamp(sp["production_start"]))].copy()   # "after go-live" months, calm period
    legit = list(MODEL_FEATURES)                                                         # the 12 approved, leakage-safe features

    print("=" * 100)
    print("ACT 1 - average_balance_leaky  (WRONG)  vs  average_balance  (RIGHT)")
    print("=" * 100)
    print("single-feature AUC on the TEST period (higher balance should mean LOWER risk, so we use -balance):")
    for name in ("average_balance", "average_balance_leaky"):
        print(f"   {name:<24} AUC = {roc_auc_score(te[TARGET], -te[name]):.4f}")
    right = _fit(tr, legit)
    wrong_feats = legit + ["average_balance_leaky"]
    wrong = _fit(tr, wrong_feats)
    rows = [("RIGHT: legit features only", _auc(right, tr, legit), _auc(right, va, legit), _auc(right, te, legit))]
    rows.append(("WRONG: + average_balance_leaky", _auc(wrong, tr, wrong_feats), _auc(wrong, va, wrong_feats), _auc(wrong, te, wrong_feats)))
    print("\nHistGradientBoosting trained on the train period (ROC-AUC):")
    print(f"   {'model':<34}{'train':>8}{'val':>8}{'test':>8}   <- the WRONG model looks fantastic in the lab")
    for r in rows:
        print(f"   {r[0]:<34}{r[1]:>8.4f}{r[2]:>8.4f}{r[3]:>8.4f}")
    # LIVE: at scoring time only the legit average_balance exists -> that is the only value we can put into the column
    live_x = live.copy()
    live_x["average_balance_leaky"] = live_x["average_balance"]
    auc_live_wrong = _auc(wrong, live_x, wrong_feats)
    auc_live_right = _auc(right, live, legit)
    print("\nGO-LIVE (calm months 2023-03 .. 2023-08, labels arrive 6 months later).")
    print("At prediction time the 'future balance' cannot be computed; the pipeline can only feed the ordinary 90-day balance:")
    print(f"   RIGHT model  live AUC = {auc_live_right:.4f}   (test was {rows[0][3]:.4f}, gap {rows[0][3] - auc_live_right:+.3f})  -> an honest, modest drop (new months, early stress)")
    print(f"   WRONG model  live AUC = {auc_live_wrong:.4f}   (test was {rows[1][3]:.4f}, gap {rows[1][3] - auc_live_wrong:+.3f})  -> FAILURE SCENARIO (a): live far below test,"
          f" and now WORSE than the honest model ({auc_live_right:.4f})")
    # variant: all history
    ah = average_balance_all_history(raw["transactions"], extract_date=pd.Timestamp("2024-12-31")).reset_index()
    t2 = te.merge(ah, on="customer_id", how="left")
    print(f"\nVariant 'all available history up to the extract date' (2024-12-31): single-feature test AUC = "
          f"{roc_auc_score(t2[TARGET], -t2['average_balance_all_history'].fillna(0)):.4f} (weaker because customers who default LATER also look bad,"
          f" but still leaky)")

    print("\n" + "=" * 100)
    print("ACT 2 - the second, subtler leak: columns that only exist because the customer already defaulted")
    print("=" * 100)
    print(f"   {'feature':<28}{'single-feature test AUC':>26}{'model(legit + feature) test AUC':>34}")
    act2 = {}
    for f in ("dpd_target_period", "recoveries_target_period", "collections_status_latest"):
        m = _fit(tr, legit + [f])
        a1, a2 = roc_auc_score(te[TARGET], te[f]), _auc(m, te, legit + [f])
        act2[f] = (a1, a2)
        print(f"   {f:<28}{a1:>26.4f}{a2:>34.4f}")
    print("   Why: default is DEFINED as 90+ days past due, so max(days_past_due in the target window) IS the label;")
    print("        recoveries and 'in_collections' are set after the default.  Nothing here is available on as_of_date.")

    print("\n" + "=" * 100)
    print("ACT 3 - the FIX: availability rule + leakage-safe pipeline + future-perturbation test + registry gate")
    print("=" * 100)
    as_of = pd.Timestamp("2022-06-30")
    safe = future_perturbation_test(lambda c, t, l, a: build_snapshot_features(c, t, l, a), raw, as_of)
    print(f"Corrupt every row not yet available at {as_of.date()} ({safe['n_future_tx_rows']:,} future transactions + unpublished loan_status), rebuild features:")
    print(f"   build_snapshot_features()   unchanged?  {safe['passed']}   changed columns: {safe['changed_columns']}")
    for name, fn in LEAKY_BUILDERS.items():
        r = future_perturbation_test(fn, raw, as_of)
        print(f"   {name:<28} unchanged?  {r['passed']}   -> {'safe' if r['passed'] else 'LEAKS (it reads the future)'}")
    print("\nAvailability rule in action (loan_status for 2022-05 is published on 2022-06-05):")
    ls = raw["loan_status"]
    row = ls[ls["month"] == pd.Timestamp("2022-05-31")].iloc[0]
    for a in (pd.Timestamp("2022-05-31"), pd.Timestamp("2022-06-30")):
        print(f"   as_of {a.date()}:  available_at {row['available_at'].date()} <= as_of ?  {row['available_at'] <= a}  -> "
              f"{'usable' if row['available_at'] <= a else 'NOT usable'}")
    from src.governance.feature_lifecycle import FeatureNotApprovedError, check_features_approved
    print("\nRegistry gate (training refuses unapproved features):")
    for names in (["salary_credit_frequency", "average_balance"], ["average_balance_leaky"]):
        try:
            check_features_approved(names)
            print(f"   {names} -> OK")
        except FeatureNotApprovedError as e:
            print(f"   {names} -> REFUSED: {e}")

    print("\n" + "=" * 100)
    print("ACT 4 - split leakage: random split vs time-based split (same legit features, same model)")
    print("=" * 100)
    pool = keep[keep["split"].isin(["train", "val", "test", "purge"])]
    rtr, rte = train_test_split(pool, test_size=0.3, random_state=0, stratify=pool[TARGET])
    m_r = _fit(rtr, legit)
    auc_random = _auc(m_r, rte, legit)
    print(f"   RANDOM split (customers appear on both sides):  test AUC = {auc_random:.4f}   <- optimistic")
    print(f"   TIME split (train earlier, test later):         test AUC = {rows[0][3]:.4f}   <- what production will look like")
    print("   Also: a random split would let the model see 2022-12 labels when predicting 2022-08 (using the future).")

    print("\nKEY TAKEAWAYS")
    print("  1. A feature that lifts test AUC from about 0.89 to 0.96 with one column is suspicious, not a triumph.")
    print("  2. Every feature needs a declared window + availability lag; code must filter by available_at <= as_of_date FIRST.")
    print("  3. Prove it with the future-perturbation test, and block unapproved features in training (registry gate).")

    import json
    from src.common.paths import REPORTS
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "leakage_demo.json").write_text(json.dumps({
        "single_feature_auc_test": {n: round(float(roc_auc_score(te[TARGET], -te[n])), 4) for n in ("average_balance", "average_balance_leaky")},
        "right_model": dict(train=rows[0][1], val=rows[0][2], test=rows[0][3], live=auc_live_right),
        "wrong_model": dict(train=rows[1][1], val=rows[1][2], test=rows[1][3], live=auc_live_wrong),
        "act2": {k: dict(single=v[0], model=v[1]) for k, v in act2.items()},
        "future_perturbation": {"build_snapshot_features": safe["passed"],
                                **{n: future_perturbation_test(fn, raw, as_of)["passed"] for n, fn in LEAKY_BUILDERS.items()}},
        "random_split_auc": auc_random, "time_split_auc": rows[0][3]}, indent=2, default=float))
    save_fact("leakage_demo", json.loads((REPORTS / "leakage_demo.json").read_text()))


if __name__ == "__main__":
    main()
