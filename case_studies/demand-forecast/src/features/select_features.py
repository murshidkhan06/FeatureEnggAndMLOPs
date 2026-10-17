"""Feature selection with the three families taught in the course - on TRAINING rows only.

   FILTER    look at each feature on its own (no model):  variance, Pearson correlation with the target,
             mutual information (MI, catches non-linear links).
   WRAPPER   RFE = Recursive Feature Elimination: fit a Ridge, throw away the weakest feature, repeat.
   EMBEDDED  Lasso: a linear model whose L1 penalty pushes useless coefficients to exactly 0.

Rule: a feature is KEPT if at least 2 of the 3 families vote for it (filter / wrapper / embedded).

Two groups, two targets (a forecasting-specific idea)
-----------------------------------------------------
Sales = LEVEL x PATTERN.   A big series sells 500 a day, a small one 50, so raw sales are dominated by the level.
  * LEVEL features   (lags, rolling means, trend_index)  are judged against  log(1 + sales)
  * PATTERN features (weekday, promo, price, weather...) are judged against the level-free target
        demand_index = log(1 + sales) - log(1 + rolling_mean_30)     ("was today above/below normal?")
    e.g. sales 120 vs recent average 100  ->  log(121) - log(101) = +0.18  (18% above normal).
Without this split, promotions/holidays would look useless simply because their effect is small next to series size.

After the votes three sensible clean-ups:
  1. Redundancy: if two kept features correlate with |r| >= 0.97, keep one.  Among LEVEL features the MORE RECENT
     one wins (lag_1 < rolling_mean_7 < lag_7 < lag_14 < rolling_mean_30), because a short window notices a change of
     level sooner - exactly what we need when demand shifts.  (rolling_mean_7 and rolling_mean_30 have r = 0.98.)
  2. Encoding rules (domain knowledge, printed in the output): the weekday is encoded ONCE (day_of_week, one-hot
     for the linear model); the time of year is encoded by month_sin / month_cos (smooth: December touches January).
  3. Absolute price in INR is NOT used as a model input: it is not comparable across products (product_id already
     identifies the product), it moves with every unit/currency change, and because list prices rise slowly it acts as a
     hidden "clock" that trees can memorise.  The relative price features (discount_pct, price_change) are kept.
Identity (product_id, store_id) and a few business-critical drivers are always kept (DOMAIN_KEEP).

Run:  python -m src.features.select_features      -> writes configs/selected_features.yaml
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import CONFIGS, DATA_RAW
from src.features.build_features import training_frame
from src.features.definitions import ALL_CANDIDATE_FEATURES, DOMAIN_KEEP, IDENTITY, LAGS, LEVEL_FEATURES
from src.features.registry import check_features_approved

WEEKDAY_FAMILY = ["day_of_week", "is_weekend", "dow_sin", "dow_cos"]
ANNUAL_FAMILY = ["month", "week_of_year", "month_sin", "month_cos"]
VOTES_NEEDED = 2
REDUNDANCY_R = 0.97
RECENCY_ORDER = ["lag_1", "rolling_mean_7", "lag_7", "lag_14", "rolling_mean_30"]   # most recent first
DROP_RULES = {"price": "absolute INR price is not comparable across products and is unit-fragile; "
                       "relative features (discount_pct, price_change) are used instead"}


def _design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric design matrix for the statistics: lags are log-transformed (multiplicative world), NaN -> median."""
    X = df[cols].copy()
    for c in cols:
        if c in LAGS:
            X[c] = np.log1p(X[c])
    return X.fillna(X.median())


def score_group(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.DataFrame:
    """One row per feature: variance flag, |r|, MI, RFE rank, Lasso coefficient and the three votes."""
    cols = list(X.columns)
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols)
    ys = (y - y.mean()) / y.std()
    # --- filter
    var_ok = VarianceThreshold(1e-3).fit(MinMaxScaler().fit_transform(X)).get_support()
    corr = Z.apply(lambda c: abs(np.corrcoef(c, ys)[0, 1]))
    sub = np.random.default_rng(seed).choice(len(X), size=min(5000, len(X)), replace=False)
    mi = pd.Series(mutual_info_regression(Z.iloc[sub], ys.iloc[sub], random_state=seed), index=cols)
    filter_vote = var_ok & ((corr.values >= 0.05) | (mi.values >= mi.median()))
    # --- wrapper (RFE with Ridge keeps ~60% of the group)
    keep_n = max(2, math.ceil(0.6 * len(cols)))
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=keep_n).fit(Z, ys)
    # --- embedded (Lasso on standardised data)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(Z, ys)
    out = pd.DataFrame(dict(variance_ok=var_ok, abs_corr=corr.round(3).values, mutual_info=mi.round(3).values,
                            rfe_rank=rfe.ranking_, lasso_coef=lasso.coef_.round(3)), index=cols)
    out["filter_vote"] = filter_vote.astype(int)
    out["wrapper_vote"] = rfe.support_.astype(int)
    out["embedded_vote"] = (np.abs(lasso.coef_) > 1e-6).astype(int)
    out["votes"] = out[["filter_vote", "wrapper_vote", "embedded_vote"]].sum(axis=1)
    return out


def run_selection(train: pd.DataFrame) -> dict:
    """Return dict(table, selected, dropped).  `train` = engineered TRAINING rows only (no validation/test!)."""
    level_cols = [c for c in LEVEL_FEATURES]
    pattern_cols = [c for c in ALL_CANDIDATE_FEATURES if c not in level_cols + IDENTITY]
    y_level = np.log1p(train["sales"])
    y_rel = np.log1p(train["sales"]) - np.log1p(train["rolling_mean_30"])
    t_level = score_group(_design(train, level_cols), y_level)
    t_level.insert(0, "group", "level  (target: log sales)")
    t_pattern = score_group(_design(train, pattern_cols), y_rel)
    t_pattern.insert(0, "group", "pattern (target: demand index)")
    table = pd.concat([t_level, t_pattern])
    table["decision"], table["reason"] = "", ""

    kept = [f for f in table.index if table.loc[f, "votes"] >= VOTES_NEEDED]
    reasons: dict[str, str] = {f: f"{table.loc[f, 'votes']}/3 votes" for f in table.index}
    dropped: dict[str, str] = {f: f"only {table.loc[f, 'votes']}/3 votes" for f in table.index if f not in kept}

    # 1) redundancy among kept features (highest votes, then MI, wins)
    D = _design(train, [f for f in kept if f in table.index])
    corr = D.corr().abs()
    order = sorted(kept, key=lambda f: (RECENCY_ORDER.index(f) if f in RECENCY_ORDER else -1,
                                        -table.loc[f, "votes"], -table.loc[f, "mutual_info"]))
    final: list[str] = []
    for f in order:
        twin = next((g for g in final if corr.loc[f, g] >= REDUNDANCY_R), None)
        if twin:
            dropped[f] = f"redundant: |r|={corr.loc[f, twin]:.3f} with {twin}"
        else:
            final.append(f)
    # 2) encoding rules
    for fam, keep_set, why in ((WEEKDAY_FAMILY, ["day_of_week"], "weekday already encoded by day_of_week (one-hot)"),
                               (ANNUAL_FAMILY, ["month_sin", "month_cos"], "annual cycle encoded by month_sin/month_cos")):
        final = [f for f in final if f not in fam or f in keep_set]
        for f in fam:
            if f not in keep_set:
                dropped[f] = why
        for f in keep_set:
            dropped.pop(f, None)
            if f not in final:
                final.append(f)
                reasons[f] = "encoding rule (domain decision)"
    for f, why in DROP_RULES.items():
        if f in final:
            final.remove(f)
        dropped[f] = why
    # 3) domain keep
    for f in DOMAIN_KEEP:
        dropped.pop(f, None)
        if f not in final:
            final.append(f)
            reasons[f] = "domain knowledge: always kept"
    for f in IDENTITY:
        reasons.setdefault(f, "series identity: always kept")
    selected = [f for f in ALL_CANDIDATE_FEATURES if f in final]
    for f in table.index:
        table.loc[f, "decision"] = "KEEP" if f in selected else "drop"
        table.loc[f, "reason"] = reasons[f] if f in selected else dropped.get(f, "")
    return dict(table=table, selected=selected, dropped={f: dropped[f] for f in dropped if f not in selected})


def load_selected() -> list[str]:
    """Selected features from configs/selected_features.yaml (falls back to all approved candidates)."""
    p = CONFIGS / "selected_features.yaml"
    if p.exists():
        return yaml.safe_load(open(p, encoding="utf-8"))["selected"]
    return list(ALL_CANDIDATE_FEATURES)


if __name__ == "__main__":
    params = load_params()
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    df = training_frame(raw)
    train = df[df["date"] <= pd.Timestamp(params["split"]["train_end"])]
    res = run_selection(train)
    pd.set_option("display.width", 200)
    print(f"Training rows used for selection: {len(train):,} (dates <= {params['split']['train_end']}; validation/test untouched)\n")
    print(res["table"].drop(columns=["variance_ok"]).to_string())
    print(f"\nSELECTED ({len(res['selected'])}): {res['selected']}")
    print("DROPPED :")
    for f, why in res["dropped"].items():
        print(f"  {f:<16} {why}")
    check_features_approved(res["selected"])                       # feature gate
    yaml.safe_dump(dict(selected=res["selected"], dropped=res["dropped"], votes_needed=VOTES_NEEDED,
                        selected_on="training rows only"), open(CONFIGS / "selected_features.yaml", "w"), sort_keys=False)
    print("\nwrote configs/selected_features.yaml  (all selected features are Approved in the registry)")
    save_fact("selection", dict(table=res["table"].reset_index(names="feature").to_dict("records"),
                                selected=res["selected"], dropped=res["dropped"]))
