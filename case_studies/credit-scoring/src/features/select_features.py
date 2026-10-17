"""Feature selection: FILTER vs WRAPPER vs EMBEDDED, then a governance-aware FINAL DECISION per feature.

WHY select features?  Fewer, better features = simpler scorecard, less overfitting, easier regulator review,
fewer things that can break in production.  But in a bank, statistics are only HALF of the decision:
a feature that predicts brilliantly may still be illegal (sensitive), unavailable at decision time (leakage)
or a proxy for a protected group.  So the final table combines both.

The three families (all computed on the TRAINING split only)
  FILTER    judge each feature alone, before any model:
              variance (is it nearly constant?)  |  ANOVA F-test (do class means differ?)
              chi-square (is the category/bin related to default?)  |  mutual information (any dependence, even non-linear)
            + correlation filter: if two features have |Spearman r| >= threshold keep only the more informative one.
  WRAPPER   RFE = Recursive Feature Elimination: fit a logistic regression, drop the weakest column, repeat.
  EMBEDDED  the model itself zeroes weak features while training: L1 (Lasso) logistic regression and Elastic Net
            (mix of L1 and L2).  Worked idea:  loss = log-loss + C^-1 * sum|coef|  -> small coefficients are pushed to exactly 0.

Votes: a feature is SELECTED when >= 2 of the 3 families support it.

Worked example of the correlation rule: |r(monthly_debt, debt_to_income_ratio)| = 0.81 >= 0.80 -> keep the one with the
higher mutual information with default, reject the other as "correlated".

Run:  python -m src.features.select_features
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.params import load_params
from src.common.paths import CONFIGS, DATA_PROCESSED, REPORTS
from src.features.definitions import ENGINEERED, EXPECTED_SIGN, FEATURE_KINDS, FRAME_COLUMNS, categories_of, kind_of
from src.features.registry import check_features_approved, load_registry
from src.models.pipeline import build_pipeline, build_preprocessor, encoded_to_feature

TARGET = "default_12m"
EXTRA_SHOWN = ["age", "gender", "marital_status", "interest_rate_offered", "loan_approved", "days_past_due_after_disbursal",
               "collections_status", "post_disbursal_payment_ratio", "credit_officer_override"]


def coded(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One numeric column per feature: numbers as-is (median-imputed), categories as integer codes."""
    out = {}
    for c in cols:
        k = kind_of(c)
        if k in ("numeric", "binary"):
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = s.fillna(s.median())
        else:
            out[c] = pd.Categorical(df[c], categories=categories_of(c)).codes
    return pd.DataFrame(out, index=df.index)


def scaled_variance(s: pd.Series) -> float:
    """Variance after clipping to the 1-99% range and min-max scaling to [0,1] (outlier-robust, scale-free)."""
    lo, hi = s.quantile([0.01, 0.99])
    if hi <= lo:
        return 0.0
    return float(((s.clip(lo, hi) - lo) / (hi - lo)).var())


def chi2_p(x: pd.Series, y: pd.Series, numeric: bool) -> tuple[float, float]:
    """Chi-square test of independence between (binned) feature and default.  Numeric features -> 5 quantile bins."""
    xs = pd.qcut(x, 5, duplicates="drop") if numeric else x
    table = pd.crosstab(xs, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    stat, p, _, _ = chi2_contingency(table)
    return float(stat), float(p)


def filter_table(df: pd.DataFrame, cols: list[str], prm: dict) -> pd.DataFrame:
    y = df[TARGET]
    X = coded(df, cols)
    discrete = np.array([kind_of(c) in ("nominal", "ordinal", "binary") for c in cols])
    mi = mutual_info_classif(X.to_numpy(float), y, discrete_features=discrete, random_state=0)
    rows = []
    for i, c in enumerate(cols):
        k = kind_of(c)
        raw = pd.to_numeric(df[c], errors="coerce") if k in ("numeric", "binary") else None
        if k in ("numeric", "binary"):
            var = scaled_variance(X[c])
            near_const = var < prm["min_scaled_variance"]
        else:
            top = df[c].value_counts(normalize=True).iloc[0]
            var, near_const = float(1 - top), bool(top > 0.99)
        groups = [X.loc[y == v, c] for v in (0, 1)]
        anova_p = float(f_oneway(*groups).pvalue) if k != "nominal" and X[c].nunique() > 1 else np.nan
        anova_f = float(f_oneway(*groups).statistic) if k != "nominal" and X[c].nunique() > 1 else np.nan
        chi, chi_p = chi2_p(X[c] if k in ("numeric", "binary") else df[c], y, numeric=(k == "numeric"))
        rows.append(dict(feature=c, kind=k, missing_pct=round(100 * df[c].isna().mean(), 1), scaled_variance=round(var, 4),
                         near_constant=near_const, anova_F=anova_f, anova_p=anova_p, chi2=chi, chi2_p=chi_p,
                         mutual_info=float(mi[i])))
    t = pd.DataFrame(rows)
    p_min = t[["anova_p", "chi2_p"]].min(axis=1)
    t["filter_pass"] = (~t.near_constant) & (p_min < prm["alpha"]) & (t.mutual_info >= prm["min_mutual_info"])
    return t


def correlation_drop(df: pd.DataFrame, cols: list[str], mi: dict, thr: float) -> tuple[dict, pd.DataFrame]:
    """Greedy rule: look at pairs from the most to the least correlated; drop the member with lower MI.

    Spearman (rank) correlation is used because it ignores the outliers that distort Pearson on income.
    If two columns are the same information (|r| >= 0.999, e.g. income and log(income)) the ENGINEERED one is kept.
    """
    num_cols = [c for c in cols if kind_of(c) in ("numeric", "binary", "ordinal")]
    corr = coded(df, num_cols).corr(method="spearman").abs()
    pairs = [(corr.loc[a, b], a, b) for i, a in enumerate(num_cols) for b in num_cols[i + 1:] if corr.loc[a, b] >= thr]
    pairs.sort(reverse=True)
    dropped: dict[str, str] = {}
    for r, a, b in pairs:
        if a in dropped or b in dropped:
            continue
        if r >= 0.999:
            loser, winner = (a, b) if (a not in ENGINEERED and b in ENGINEERED) else (b, a) if (b not in ENGINEERED and a in ENGINEERED) else (a, b)
        else:
            loser, winner = (a, b) if mi[a] < mi[b] else (b, a)
        dropped[loser] = f"correlated with {winner} (|r|={r:.2f})"
    pair_df = pd.DataFrame([(a, b, round(float(r), 3)) for r, a, b in pairs], columns=["feature_a", "feature_b", "abs_spearman"])
    return dropped, pair_df


def _sk_version() -> tuple[int, int]:
    import sklearn
    a = sklearn.__version__.split(".")
    return int(a[0]), int(a[1])


def l1_logreg(C: float) -> LogisticRegression:
    """Lasso-style logistic regression.  (scikit-learn >= 1.8 spells it l1_ratio=1; older versions penalty='l1'.)"""
    if _sk_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=5000, random_state=0)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)


def elasticnet_logreg(C: float, l1_ratio: float) -> LogisticRegression:
    """Elastic Net logistic regression = a mix of L1 (zeroes features) and L2 (shrinks them)."""
    if _sk_version() >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, solver="saga", C=C, max_iter=5000, random_state=0)
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C, max_iter=5000, random_state=0)


def wrapper_embedded(df: pd.DataFrame, features: list[str], prm: dict) -> pd.DataFrame:
    """RFE + L1 + ElasticNet on the one-hot/standardised design matrix; results are folded back to ORIGINAL features."""
    y = df[TARGET]
    prep = Pipeline([("prep", build_preprocessor(features)), ("scale", StandardScaler())]).fit(df[features], y)
    Z = prep.transform(df[features])
    names = list(prep.named_steps["prep"].get_feature_names_out())
    owner = [encoded_to_feature(n, features) for n in names]

    n_keep = max(1, int(round(prm["rfe_keep_fraction"] * Z.shape[1])))
    rfe = RFE(LogisticRegression(C=0.5, max_iter=2000), n_features_to_select=n_keep, step=1).fit(Z, y)
    l1 = l1_logreg(prm["lasso_C"]).fit(Z, y)
    en = elasticnet_logreg(prm["enet_C"], prm["enet_l1_ratio"]).fit(Z, y)
    t = pd.DataFrame(dict(owner=owner, rfe_keep=rfe.support_, rfe_rank=rfe.ranking_, l1=l1.coef_[0], enet=en.coef_[0]))
    g = t.groupby("owner").agg(rfe_selected=("rfe_keep", "any"), rfe_best_rank=("rfe_rank", "min"),
                               lasso_max_abs_coef=("l1", lambda s: float(np.abs(s).max())),
                               enet_max_abs_coef=("enet", lambda s: float(np.abs(s).max())))
    g["lasso_nonzero"] = g.lasso_max_abs_coef > 1e-8
    g["enet_nonzero"] = g.enet_max_abs_coef > 1e-8
    return g.reset_index().rename(columns={"owner": "feature"})


def sign_check(train: pd.DataFrame, selected: list[str]) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    """Business-sense check on the shortlist: remove features whose fitted coefficient has the WRONG sign.

    Example: if `emi_to_income` gets a NEGATIVE coefficient (bigger instalment share = safer?) while its sibling
    `loan_to_income` carries the real effect, the model is exploiting multicollinearity.  Such a coefficient cannot be
    explained to a regulator or an applicant, so the feature is removed.  One violator per round (the strongest), refit, repeat.
    """
    selected, removed, log = list(selected), {}, []
    while True:
        pipe = build_pipeline(selected).fit(train[selected], train[TARGET])
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=[encoded_to_feature(n, selected) for n in names])
        single = coef.groupby(level=0).sum()
        bad = {f: c for f, c in single.items() if f in EXPECTED_SIGN and np.sign(c) == -EXPECTED_SIGN[f] and abs(c) > 0.02}
        log += [dict(round=len(removed) + 1, feature=f, coef=round(float(c), 3), expected=("+" if EXPECTED_SIGN[f] > 0 else "-")) for f, c in single.items() if f in EXPECTED_SIGN]
        if not bad:
            break
        worst = max(bad, key=lambda f: abs(bad[f]))
        removed[worst] = f"counter-intuitive sign (coef {bad[worst]:+.2f}, expected {'+' if EXPECTED_SIGN[worst] > 0 else '-'}): multicollinearity"
        selected.remove(worst)
    return selected, removed, pd.DataFrame(log)


def run(train: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    prm = {**load_params()["selection"]}
    reg = load_registry()
    train = pd.read_csv(DATA_PROCESSED / "train.csv") if train is None else train
    eligible = [c for c in FRAME_COLUMNS if reg[c]["decision_category"] == "ELIGIBLE"]
    shown = FRAME_COLUMNS + [c for c in EXTRA_SHOWN if c not in FRAME_COLUMNS]

    ft = filter_table(train, shown, prm)
    mi = dict(zip(ft.feature, ft.mutual_info))
    dropped_corr, pairs = correlation_drop(train, eligible, mi, prm["corr_threshold"])
    survivors = [c for c in eligible if c not in dropped_corr and not bool(ft.set_index("feature").loc[c, "near_constant"])]
    we = wrapper_embedded(train, survivors, prm)
    t = ft.merge(we, on="feature", how="left")

    decisions, votes_col = [], []
    for _, r in t.iterrows():
        f, cat = r.feature, reg[r.feature]["decision_category"]
        votes = int(r.filter_pass) + int(bool(r.rfe_selected)) + int(bool(r.lasso_nonzero)) if f in survivors else np.nan
        votes_col.append(votes)
        if cat == "LEAKAGE":
            d = "REJECTED (leakage)"
        elif cat == "UNAVAILABLE_AT_PREDICTION":
            d = "REJECTED (not available at prediction time)"
        elif cat == "SENSITIVE_POLICY":
            d = "REJECTED (sensitive, policy)"
        elif cat == "REQUIRES_GOVERNANCE":
            d = "REQUIRES GOVERNANCE (default: excluded)"
        elif f in dropped_corr:
            d = f"REJECTED (correlated: {dropped_corr[f]})"
        elif f not in survivors:
            d = "REJECTED (near-constant)"
        elif votes >= prm["min_votes"]:
            d = "SELECTED"
        else:
            d = f"REJECTED (weak signal: {int(votes)}/3 votes)"
        decisions.append(d)
    t["votes"], t["decision"] = votes_col, decisions
    shortlist = [f for f, d in zip(t.feature, t.decision) if d == "SELECTED"]
    selected, sign_removed, sign_log = sign_check(train, shortlist)
    t["decision"] = [f"REJECTED ({sign_removed[f]})" if f in sign_removed else d for f, d in zip(t.feature, t.decision)]
    check_features_approved(selected, reg)                       # a selected feature MUST be approved by governance

    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "feature_selection.csv", index=False)
    pairs.to_csv(REPORTS / "correlated_pairs.csv", index=False)
    (CONFIGS / "selected_features.yaml").write_text(
        "# Written by `python -m src.features.select_features` - the features the production model uses.\n"
        + yaml.safe_dump({"selected_features": selected,
                          "rejected": {f: d for f, d in zip(t.feature, t.decision) if d != "SELECTED"}}, sort_keys=False, width=140),
        encoding="utf-8")
    result = dict(selected=selected, table=t, pairs=pairs, dropped_corr=dropped_corr, params=prm, sign_removed=sign_removed, sign_log=sign_log)
    if verbose:
        print_report(result)
    return result


def print_report(res: dict) -> None:
    t, pairs = res["table"], res["pairs"]
    pd.set_option("display.width", 250)
    print("=== 1. CORRELATION FILTER (Spearman, eligible features, threshold "
          f"{res['params']['corr_threshold']}) ===")
    print(pairs.to_string(index=False) if len(pairs) else "no pairs above threshold")
    print("\n=== 2. FILTER statistics on TRAIN (all columns, including the ones we will reject) ===")
    f = t[["feature", "kind", "missing_pct", "scaled_variance", "anova_F", "anova_p", "chi2_p", "mutual_info", "filter_pass"]].copy()
    f["anova_F"], f["anova_p"], f["chi2_p"] = f.anova_F.round(1), f.anova_p.map(lambda v: f"{v:.1e}"), f.chi2_p.map(lambda v: f"{v:.1e}")
    f["mutual_info"] = f.mutual_info.round(4)
    print(f.sort_values("mutual_info", ascending=False).to_string(index=False))
    print("\n  WARNING: leakage columns have by far the largest scores. 'Too good to be true' is the signal to ask:")
    print("  'would this information exist at the moment of the decision?'")
    print("\n=== 3. WRAPPER (RFE) and EMBEDDED (L1 / Elastic Net) on the features that survived the correlation filter ===")
    w = t[t.rfe_selected.notna()][["feature", "rfe_selected", "rfe_best_rank", "lasso_max_abs_coef", "lasso_nonzero", "enet_max_abs_coef", "enet_nonzero"]].copy()
    w["lasso_max_abs_coef"], w["enet_max_abs_coef"] = w.lasso_max_abs_coef.round(3), w.enet_max_abs_coef.round(3)
    print(w.to_string(index=False))
    print("\n=== 4. FINAL DECISION PER FEATURE (statistics + governance) ===")
    final = t[["feature", "filter_pass", "rfe_selected", "lasso_nonzero", "votes", "decision"]].copy()
    final["rank"] = final.decision.map(lambda d: 0 if d == "SELECTED" else 1 if d.startswith("REJECTED (corr") or d.startswith("REJECTED (weak") or "near" in d else 2)
    print(final.sort_values(["rank", "feature"]).drop(columns="rank").to_string(index=False))
    print("\n=== 5. SIGN / BUSINESS-SENSE CHECK on the shortlist ===")
    if res["sign_removed"]:
        for f, why in res["sign_removed"].items():
            print(f"   removed {f}: {why}")
    else:
        print("   all coefficient signs agree with business expectations")
    print(f"\nSELECTED ({len(res['selected'])}): {res['selected']}")


if __name__ == "__main__":
    run()
