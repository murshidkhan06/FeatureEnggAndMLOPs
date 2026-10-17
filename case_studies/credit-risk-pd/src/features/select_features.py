"""Feature SELECTION: filter vs wrapper vs embedded, using TRAINING snapshots only.

WHY select at all?  Fewer features = a simpler model that is easier to explain to a regulator, cheaper to compute in production
(every feature is a data pipeline someone must maintain) and less likely to overfit.

THREE FAMILIES (each answers "which features matter?" in a different way)
  FILTER    looks at each feature alone (or in pairs), before any model is trained.
              * variance        a feature that never changes carries no information
              * ANOVA F-test    do defaulters and non-defaulters have different means?  (small p-value = yes)
              * mutual info     any statistical dependence with the target, also non-linear
              * correlation     if two features are |r| > 0.80, keep only the one with the higher mutual information
  WRAPPER   RFE (recursive feature elimination): fit a model, delete the weakest feature, repeat until n_rfe remain.
  EMBEDDED  L1-penalised logistic regression (Lasso-style): the penalty  C-controlled  sum |b|  pushes useless weights to exactly 0.
              Penalised loss = log-loss + (1/C) * sum(|b_j|)    smaller C = stronger penalty = more zeros.

HOW the final list is decided: a feature is kept if at least `min_votes` (=2) of the 3 families keep it.
THEN AN OUT-OF-TIME VETO: selection saw only TRAIN rows, and the same customer appears in many monthly snapshots, so p-values are
too optimistic and a static customer attribute (account_age) can look useful by chance (it acts like a customer fingerprint).  We refit the
baseline on TRAIN and measure permutation importance on the later VALIDATION period; a feature whose shuffling IMPROVES validation
ROC-AUC by more than `oot_veto_below` is removed.  (Test data is never used.)
Finally check_features_approved() makes sure every kept feature is Approved + Production in the registry.

CAVEATS
  * Selection is done on TRAIN snapshots only (same rows the model is fitted on) - selecting on test data would leak.
  * Correlated features (missed_payment_count vs loan_payment_ratio) share credit: a filter may drop one although it is useful.
  * Statistical association is not causation; "kept" means "useful for prediction", not "causes default".

Run:  python -m src.features.select_features
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import RFE, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.common.config import load_params
from src.common.facts import save_fact
from src.common.paths import REPORTS, SELECTED_FEATURES_PATH, SNAPSHOTS_PATH
from src.features.build_features import add_derived_features
from src.features.definitions import MODEL_FEATURES, TARGET
from src.governance.feature_lifecycle import check_features_approved


warnings.filterwarnings("ignore", category=FutureWarning)      # sklearn 1.8 deprecates the penalty= argument of LogisticRegression
warnings.filterwarnings("ignore", category=UserWarning)


def run_selection(train: pd.DataFrame, features: list[str], p: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (the comparison table, the selected feature list).  `train` must contain ONLY training rows."""
    y = train[TARGET].to_numpy()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(train[features]), columns=features)  # median fitted on TRAIN
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)
    t = pd.DataFrame(index=features)
    t["missing_%"] = (train[features].isna().mean() * 100).round(1)
    # ---- FILTER ---------------------------------------------------------------------------------------------------
    t["variance_scaled"] = MinMaxScaler().fit_transform(X).var(axis=0).round(4)     # variance after 0-1 scaling (comparable)
    keep_var = VarianceThreshold(p["min_variance"]).fit(MinMaxScaler().fit_transform(X)).get_support()
    F, pval = f_classif(X, y)
    t["anova_F"], t["anova_p"] = F.round(1), pval
    t["mutual_info"] = mutual_info_classif(X, y, random_state=0).round(4)
    corr = X.corr().abs()
    dropped_corr: dict[str, str] = {}
    for a in features:                                     # correlation pruning: weaker (lower MI) of a correlated pair goes
        for b in features:
            if a < b and corr.loc[a, b] > p["max_abs_corr"] and a not in dropped_corr and b not in dropped_corr:
                weaker = a if t.loc[a, "mutual_info"] < t.loc[b, "mutual_info"] else b
                dropped_corr[weaker] = f"|r|={corr.loc[a, b]:.2f} with {b if weaker == a else a}"
    t["corr_dropped"] = [dropped_corr.get(f, "") for f in features]
    t["FILTER_keep"] = keep_var & (pval < 0.01) & np.array([f not in dropped_corr for f in features])
    # ---- WRAPPER (RFE) --------------------------------------------------------------------------------------------
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=p["n_rfe"]).fit(Xs, y)
    t["rfe_rank"] = rfe.ranking_                       # 1 = kept
    t["WRAPPER_keep"] = rfe.support_
    # ---- EMBEDDED (L1) --------------------------------------------------------------------------------------------
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=p["l1_C"], max_iter=2000).fit(Xs, y)
    t["l1_coef"] = l1.coef_[0].round(3)
    t["EMBEDDED_keep"] = l1.coef_[0] != 0
    t["votes"] = t[["FILTER_keep", "WRAPPER_keep", "EMBEDDED_keep"]].sum(axis=1)
    t["SELECTED"] = t["votes"] >= p["min_votes"]
    return t, t.index[t["SELECTED"]].tolist()


def oot_veto(train: pd.DataFrame, val: pd.DataFrame, selected: list[str], params: dict) -> pd.Series:
    """Permutation importance (ROC-AUC drop) of each selected feature on the VALIDATION period, for the baseline fitted on TRAIN.

    Negative = the model does BETTER on later data when this column is scrambled -> the feature learned something that does not last."""
    from sklearn.inspection import permutation_importance
    from src.models.pipelines import make_logistic
    pipe = make_logistic(selected, params["model"]["logistic"]).fit(train[selected], train[TARGET])
    r = permutation_importance(pipe, val[selected], val[TARGET], scoring="roc_auc", n_repeats=10, random_state=0)
    return pd.Series(r.importances_mean, index=selected).round(4)


def main() -> None:
    params = load_params()
    snap = add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
    train = snap[snap["split"] == "train"]
    check_features_approved(MODEL_FEATURES)              # candidates must be registered + approved before they are even compared
    table, selected = run_selection(train, MODEL_FEATURES, params["selection"])
    val = snap[snap["split"] == "val"]
    oot = oot_veto(train, val, selected, params)
    veto = oot.index[oot < -params["selection"]["oot_veto_below"]].tolist()
    table["oot_perm_imp_val"] = oot
    table["OOT_VETO"] = table.index.isin(veto)
    table["SELECTED"] = table["SELECTED"] & ~table["OOT_VETO"]
    selected = table.index[table["SELECTED"]].tolist()
    pd.set_option("display.width", 250)
    show = table.copy()
    show["anova_p"] = show["anova_p"].map(lambda v: f"{v:.1e}")
    for c in ("FILTER_keep", "WRAPPER_keep", "EMBEDDED_keep", "OOT_VETO", "SELECTED"):
        show[c] = show[c].map({True: "KEEP", False: "-"})
    print(f"Training rows: {len(train):,}  (as_of {train['as_of_date'].min().date()} .. {train['as_of_date'].max().date()}), "
          f"default rate {train[TARGET].mean():.2%}\n")
    print(show.sort_values(["votes", "mutual_info"], ascending=False).to_string())
    print("\nFilter drops for correlation:", {k: v for k, v in zip(table.index, table["corr_dropped"]) if v} or "none")
    print(f"\nOUT-OF-TIME VETO (validation permutation importance < -{params['selection']['oot_veto_below']}): {veto or 'none'}")
    print(f"\nSELECTED ({len(selected)} of {len(MODEL_FEATURES)}; kept by >= {params['selection']['min_votes']} of 3 families, minus the out-of-time veto):\n  {selected}")
    not_sel = [f for f in MODEL_FEATURES if f not in selected]
    print(f"NOT selected: {not_sel}")
    out = {"selected": selected, "not_selected": not_sel, "oot_vetoed": veto,
           "rule": f"kept by >= {params['selection']['min_votes']} of 3 families (filter, wrapper, embedded) on TRAIN snapshots, "
                   "then vetoed if its validation permutation importance is clearly negative",
           "votes": {f: int(v) for f, v in table["votes"].items()}}
    SELECTED_FEATURES_PATH.write_text("# Output of: python -m src.features.select_features  (re-generated by the DVC 'select' stage)\n"
                                      + yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "feature_selection.csv")
    save_fact("selection", dict(selected=selected, not_selected=not_sel, oot_vetoed=veto, corr_dropped={k: v for k, v in zip(table.index, table["corr_dropped"]) if v},
                                table=table.reset_index(names="feature")[["feature", "missing_%", "anova_F", "mutual_info", "FILTER_keep", "rfe_rank", "WRAPPER_keep",
                                                                           "l1_coef", "EMBEDDED_keep", "votes", "oot_perm_imp_val", "OOT_VETO", "SELECTED"]].to_dict("records")))
    print(f"\nwrote {SELECTED_FEATURES_PATH.name}")


if __name__ == "__main__":
    main()
