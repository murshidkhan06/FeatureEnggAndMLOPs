"""Data-leakage demo: the SAME model, once with a forbidden column / junior-engineer mistake, once done right.

Golden question for every feature:  "Would this information genuinely be available at the moment of the decision?"

Part A  Leaky columns.  interest_rate_offered is set AFTER risk scoring; days_past_due_after_disbursal exists only months
        after the loan was paid out.  Adding them makes the offline AUC look fantastic - and the model is useless in
        production (the column is empty or does not exist when a real application arrives).
Part B  Three junior-engineer mistakes (all produce numbers that are too good):
          B1  scaler / imputer fitted on ALL data before the split          (leakage through preprocessing)
          B2  target encoding computed with the labels of the TEST rows       (leakage through a target-derived feature)
          B3  random row split when the same applicant appears twice          (applicant / temporal leakage)

Run:  python -m src.features.leakage_demo
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src import project as P
from src.common.paths import DATA_PROCESSED, REPORTS
from src.features.registry import FeatureNotApproved, check_features_approved, load_selected
from src.models.pipeline import build_pipeline, build_preprocessor, fit_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEG = ["location", "employment_type", "education", "loan_term_months"]          # segment used for the (mis)used target encoding


def target_encode_fit(df: pd.DataFrame, y: pd.Series, m: float = 20.0) -> tuple[pd.Series, float]:
    """Smoothed mean default rate per segment:  (sum_y + m*global_rate) / (n + m).   m = pseudo-count (shrinks small groups).
    Tiny example: segment with n=4 rows, 2 defaults, global rate 0.10, m=20 -> (2 + 2)/(4+20) = 0.167 (not 0.50).
    m=0 is the plain average (the junior's version): a segment with ONE row gets that row's own label as its feature."""
    g = float(y.mean())
    key = df[SEG].astype(str).agg("|".join, axis=1)
    s = y.groupby(key).agg(["sum", "count"])
    return (s["sum"] + m * g) / (s["count"] + m), g


def target_encode_apply(df: pd.DataFrame, table: pd.Series, g: float) -> np.ndarray:
    key = df[SEG].astype(str).agg("|".join, axis=1)
    return key.map(table).fillna(g).to_numpy()


def auc_of(model, Xtr, ytr, Xte, yte) -> float:
    model.fit(Xtr, ytr)
    return float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))


def main() -> None:
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    valid = pd.read_csv(DATA_PROCESSED / "valid.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    feats = load_selected()
    out: dict = {}

    print("=" * 100 + "\nPART A - LEAKY COLUMNS (same model, same time-based split, only the feature list changes)\n" + "=" * 100)
    rows = []
    for name, extra in [("HONEST: approved features only", []),
                        ("+ interest_rate_offered", ["interest_rate_offered"]),
                        ("+ days_past_due_after_disbursal", ["days_past_due_after_disbursal"]),
                        ("+ both leaky columns", ["interest_rate_offered", "days_past_due_after_disbursal"])]:
        _, mv, mt = fit_score(feats + extra, "logreg", 0.5, train, valid, test)
        rows.append(dict(model=name, valid_auc=mv["roc_auc"], test_auc=mt["roc_auc"], test_ks=mt["ks_stat"]))
    a = pd.DataFrame(rows)
    print(a.to_string(index=False))
    out["part_a"] = rows
    print("\nThe registry blocks this at training time:")
    try:
        check_features_approved(feats + ["interest_rate_offered", "days_past_due_after_disbursal"])
    except FeatureNotApproved as exc:
        print(str(exc))

    print("\nWhat happens in PRODUCTION: the column does not exist yet, so the pipeline receives a missing value (median-imputed).")
    leaky_feats = feats + ["interest_rate_offered"]
    pipe, _, _ = fit_score(leaky_feats, "logreg", 0.5, train, valid, test)
    prod = test.copy()
    honest_auc = roc_auc_score(test[P.TARGET], pipe.predict_proba(test[leaky_feats])[:, 1])
    prod["interest_rate_offered"] = np.nan
    prod_auc = roc_auc_score(prod[P.TARGET], pipe.predict_proba(prod[leaky_feats])[:, 1])
    print(f"   leaky model AUC offline (column present) = {honest_auc:.3f}   ->   in production (column missing) = {prod_auc:.3f}")
    print("   (below 0.5 is possible: the model learned to 'correct' the leaky column using the honest columns; remove the column and the corrections point the wrong way)")
    out["part_a_production_collapse"] = dict(offline_auc=round(honest_auc, 4), production_auc=round(prod_auc, 4))

    print("\n" + "=" * 100 + "\nPART B - JUNIOR-ENGINEER MISTAKES\n" + "=" * 100)
    b = []
    # ---- B1 preprocessing fitted on everything -------------------------------------------------------------
    allX = pd.concat([train[feats], valid[feats], test[feats]])
    lr = LogisticRegression(C=0.5, max_iter=2000)
    right = Pipeline([("prep", build_preprocessor(feats)), ("scale", StandardScaler()), ("clf", lr)])
    auc_right = auc_of(right, train[feats], train[P.TARGET], test[feats], test[P.TARGET])
    pre = Pipeline([("prep", build_preprocessor(feats)), ("scale", StandardScaler())]).fit(allX)          # WRONG: sees test rows
    auc_wrong = auc_of(LogisticRegression(C=0.5, max_iter=2000), pre.transform(train[feats]), train[P.TARGET], pre.transform(test[feats]), test[P.TARGET])
    b.append(dict(mistake="B1 imputer+scaler fitted on ALL data", wrong_test_auc=round(auc_wrong, 4), right_test_auc=round(auc_right, 4)))

    # ---- B2 target encoding with test labels ------------------------------------------------------------------
    full = pd.concat([train, valid, test], ignore_index=True)
    pre_r = Pipeline([("prep", build_preprocessor(feats)), ("scale", StandardScaler())]).fit(train[feats])       # train-only (right)
    tab_all, g_all = target_encode_fit(full, full[P.TARGET], m=0)                                             # WRONG: plain mean incl. test labels
    wrong_tr = np.c_[pre_r.transform(train[feats]), target_encode_apply(train, tab_all, g_all)]
    wrong_te = np.c_[pre_r.transform(test[feats]), target_encode_apply(test, tab_all, g_all)]
    auc_wrong2 = auc_of(LogisticRegression(C=0.5, max_iter=2000), wrong_tr, train[P.TARGET], wrong_te, test[P.TARGET])
    # RIGHT: out-of-fold encoding for train rows, train-only table for test rows
    oof = np.zeros(len(train))
    for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(train):
        t_, g_ = target_encode_fit(train.iloc[tr_i], train[P.TARGET].iloc[tr_i])
        oof[va_i] = target_encode_apply(train.iloc[va_i], t_, g_)
    t_tr, g_tr = target_encode_fit(train, train[P.TARGET])
    right_tr = np.c_[pre_r.transform(train[feats]), oof]
    right_te = np.c_[pre_r.transform(test[feats]), target_encode_apply(test, t_tr, g_tr)]
    auc_right2 = auc_of(LogisticRegression(C=0.5, max_iter=2000), right_tr, train[P.TARGET], right_te, test[P.TARGET])
    nseg = full[SEG].astype(str).agg("|".join, axis=1).nunique()
    b.append(dict(mistake="B2 target encoding of segment (uses test labels)", wrong_test_auc=round(auc_wrong2, 4), right_test_auc=round(auc_right2, 4)))
    print(f"   (B2 segment = {SEG}: {nseg} segments for {len(full)} rows, about {len(full)//nseg} rows each -> a test row's own label is "
          f"~1/{len(full)//nseg} of its segment mean; tiny segments leak the label almost completely)")

    # ---- B3 random row split vs applicant-level split on the SAME rows (isolates the duplicate-applicant effect) ----
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    def memoriser():
        return Pipeline([("prep", build_preprocessor(feats)), ("scale", StandardScaler()),
                         ("clf", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.2, max_leaf_nodes=63,
                                                                min_samples_leaf=2, random_state=0))])
    overlaps, res = [], {"logistic regression": ([], []), "flexible boosting (can memorise)": ([], [])}
    for seed in range(3):
        for kind, splitter in (("random", ShuffleSplit(1, test_size=0.15, random_state=seed)),
                               ("applicant", GroupShuffleSplit(1, test_size=0.15, random_state=seed))):
            tr_i, te_i = next(splitter.split(full, groups=full.applicant_id if kind == "applicant" else None))
            a_, b_ = full.iloc[tr_i], full.iloc[te_i]
            if kind == "random":
                overlaps.append(b_.applicant_id.isin(a_.applicant_id).mean())
            for name, fn in (("logistic regression", lambda: build_pipeline(feats, "logreg")), ("flexible boosting (can memorise)", memoriser)):
                res[name][0 if kind == "random" else 1].append(auc_of(fn(), a_[feats], a_[P.TARGET], b_[feats], b_[P.TARGET]))
    overlap = float(np.mean(overlaps))
    for name, (w, r) in res.items():
        b.append(dict(mistake=f"B3 random row split, {name} (mean of 3 splits)", wrong_test_auc=round(float(np.mean(w)), 4), right_test_auc=round(float(np.mean(r)), 4)))
    print(f"   (B3 random split: {overlap:.1%} of test applications belong to an applicant who is ALSO in train; "
          "'right' = split by applicant_id, same rows)")
    # time-based vs random split (second leak: using the future to predict the past)
    w_t = auc_of(build_pipeline(feats, "logreg"), t_tr := pd.concat([train, valid])[feats], pd.concat([train, valid])[P.TARGET], test[feats], test[P.TARGET])
    print(f"   (reference: honest TIME-based split, logistic regression test AUC = {w_t:.4f}; in this synthetic world the process is stable over time, "
          "so a random split does not inflate much - in real lending data with seasonality / policy changes it does)")
    bt = pd.DataFrame(b)
    bt["inflation"] = (bt.wrong_test_auc - bt.right_test_auc).round(4)
    print("\n" + bt.to_string(index=False))
    out["part_b"] = bt.to_dict("records")
    out["b3_overlap_share"] = round(float(overlap), 4)
    print("\nTakeaways\n"
          " * A too-good AUC (0.95+) in lending is a SMELL, not a success: ask 'when does this column exist?'\n"
          " * Fit imputers / scalers / encoders on TRAIN only (inside the Pipeline).  Effect is small for scaling, large for target-derived features.\n"
          " * Split by TIME and by APPLICANT; never by random row when customers can apply more than once.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "leakage_demo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
