"""EXPLAINABILITY: which inputs does the champion lean on?   python -m src.models.explain

Three views (and what each can and cannot tell you):
  1. PERMUTATION IMPORTANCE (core)  - shuffle ONE input column on the TEST period and measure how much the MAE gets worse.
        importance = MAE(shuffled) - MAE(original)        a big increase = the model needed that column.
        Worked example: MAE 15.6 -> 40.1 after shuffling lag_1  =>  lag_1 importance = 24.5 units.
        Caveat: correlated columns (lag_1, rolling_mean_7, lag_7 move together) SHARE the credit: shuffling one is partly
        covered by the others, so each looks less important than the group really is.
  2. LINEAR VIEW (core)  - standardised coefficients of a Ridge-on-log model (the 'ridge_log' candidate), trained on TRAIN only.
        coefficient x (feature scaled to mean 0, std 1) = the change in log(units) for a +1 std move of that feature.
        e.g. coefficient 0.30 on log(lag_1): +1 std of log(lag_1) -> units x exp(0.30) = +35%.   Only meaningful for scaled inputs.
  3. SHAP (OPTIONAL)  - average |contribution| per column from shap's model-agnostic explainer on 150 test rows.  Skipped quietly
        if the shap package is missing or slow.   (OPTIONAL: not needed for any core step.)

ASSOCIATION IS NOT CAUSATION.  "promotion_flag is important" says the model uses it; it does not prove a promotion CAUSES the
extra units (promotions are also planned for strong weeks).  Importance describes the MODEL, not the world.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.artifact import load_artifact
from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, DATA_RAW, REPORTS
from src.features.build_features import training_frame
from src.models.pipelines import build_pipeline
from src.models.train import score, time_split


def permutation_table(pipe, frame: pd.DataFrame, features: list[str], seed: int = 0, repeats: int = 5) -> pd.DataFrame:
    """MAE increase when one column is shuffled (mean and std over `repeats` shuffles)."""
    r = permutation_importance(pipe, frame[features], frame["sales"], scoring="neg_mean_absolute_error",
                               n_repeats=repeats, random_state=seed)
    return (pd.DataFrame(dict(feature=features, mae_increase=r.importances_mean, std=r.importances_std))
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def linear_view(train: pd.DataFrame, features: list[str], params: dict) -> pd.DataFrame:
    """Standardised coefficients of the ridge_log model (log scale)."""
    pipe = build_pipeline("ridge_log", features, params).fit(train[features], train["sales"])
    names = pipe[:-1].get_feature_names_out()
    coef = pipe[-1].ridge_.coef_
    return pd.DataFrame(dict(column=names, coefficient=coef, abs_coefficient=np.abs(coef))).sort_values("abs_coefficient", ascending=False)


def shap_table(pipe, frame: pd.DataFrame, features: list[str], n: int = 100) -> pd.DataFrame | None:   # OPTIONAL
    """Mean |SHAP value| per ENCODED column (one-hot columns appear separately).  We explain the final model on the
    pre-processed numbers (pipe[:-1] = the fitted preprocessing), which keeps SHAP simple and all-numeric."""
    try:
        import shap
        X = np.asarray(pipe[:-1].transform(frame[features]), dtype=float)
        names = list(pipe[:-1].get_feature_names_out())
        rng = np.random.default_rng(0)
        bg, rows = X[rng.choice(len(X), 50, replace=False)], X[rng.choice(len(X), n, replace=False)]
        sv = shap.Explainer(pipe[-1].predict, bg)(rows, max_evals=2 * X.shape[1] + 1, silent=True)
        return (pd.DataFrame(dict(feature=names, mean_abs_shap=np.abs(sv.values).mean(axis=0)))
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    except Exception as exc:  # noqa: BLE001
        print(f"   (SHAP skipped: {type(exc).__name__}: {str(exc)[:80]})")
        return None


def main() -> None:
    art, params = load_artifact(CHAMPION_PATH), load_params()
    feats = art["input_columns"]
    df = training_frame(pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"]))
    train, valid, test = time_split(df, params["split"])
    base_mae = float(np.mean(np.abs(test["sales"] - score(art["pipeline"], test, feats))))
    print(f"CHAMPION {art['version']} ({art['extra']['model_name']}); TEST period {test.date.min().date()}..{test.date.max().date()}, MAE = {base_mae:.2f} units\n")
    print("1) PERMUTATION IMPORTANCE on TEST (MAE increase in units when the column is shuffled)")
    pt = permutation_table(art["pipeline"], test, feats)
    print(pt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    grp = pt[pt.feature.isin(["lag_1", "lag_7", "lag_14", "rolling_mean_7"])]["mae_increase"].sum()
    print(f"\n   The four level features (lag_1, lag_7, lag_14, rolling_mean_7) together: {grp:.2f} units.  They are strongly correlated,")
    print("   so each one alone understates how much the model depends on 'the recent level of this series'.")
    print("\n2) LINEAR VIEW: standardised coefficients of ridge_log (train period only; log scale)")
    lv = linear_view(pd.concat([train]), feats, params)
    print(lv.head(10).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("   Positive = more units.  The coefficients of log(lag) columns say 'the recent level dominates' (the same message as above).")
    print("\n3) SHAP (OPTIONAL)")
    sh = shap_table(art["pipeline"], test, feats)
    if sh is not None:
        print(sh.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nREMINDER: importance = what the MODEL uses (association).  It does not prove that a feature CAUSES sales.")
    REPORTS.mkdir(exist_ok=True)
    save_fact("explain", dict(model=art["extra"]["model_name"], base_mae=base_mae, permutation=pt.head(10).round(4).to_dict("records"),
                              linear=lv.head(10).round(4).to_dict("records"), shap=None if sh is None else sh.head(8).round(4).to_dict("records")))


if __name__ == "__main__":
    main()
