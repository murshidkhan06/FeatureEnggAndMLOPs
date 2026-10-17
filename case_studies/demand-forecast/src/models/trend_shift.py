"""THE teaching moment: what happens to each model when the level of the business moves above anything seen in training?

Run:  python -m src.models.trend_shift        (prints tables, saves reports/trend_shift_extrapolation.png)

Why can a tree not extrapolate?
  A tree (and a forest, and gradient boosting) ends in LEAVES, and each leaf stores the AVERAGE of the training targets that
  fell into it.  So every prediction is an average of numbers seen in training - it can never be larger than the largest
  leaf.  If a series that used to sell ~100 units a day now sells ~200, the lag_1 = 200 lands in the 'highest lag' leaf,
  which holds targets of ~120-150, and the forecast stays too low.
  A linear model computes  prediction = intercept + sum(weight x feature)  - bigger inputs give bigger outputs.
  Ours works on log(lags): doubling the lags adds log 2 to the log forecast, i.e. roughly doubles it.

Four experiments
  A. WAPE and bias of every model on three production batches (normal / drifted / festival)
  B. weekly bias on the DRIFTED batch (the shift arrives in four steps)
  C. STRESS TEST: multiply the series (sales and lags) by k = 1 .. 3.  A perfect scale-free model keeps bias ~ 0.
  D. ABLATION: give the tree a `trend_index` clock.  It helps on THIS shift, but only because the shift keeps going up.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.facts import save_fact  # noqa: E402
from src.common.params import load_params  # noqa: E402
from src.common.paths import DATA_PRODUCTION, DATA_RAW, REPORTS  # noqa: E402
from src.features.build_features import features_for_batch, training_frame  # noqa: E402
from src.features.select_features import load_selected  # noqa: E402
from src.models import metrics as M  # noqa: E402
from src.models.baselines import BASELINES  # noqa: E402
from src.models.pipelines import build_pipeline  # noqa: E402
from src.models.train import score, time_split  # noqa: E402

LEVEL_COLS = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]


def load_batch(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(DATA_PRODUCTION / f"batch_{name}.csv")
    f = features_for_batch(b, raw)
    return f[f["sales"].notna()].reset_index(drop=True)


def fit_models(df: pd.DataFrame, params: dict, feats: list[str]) -> dict:
    tr, va, _ = time_split(df, params["split"])
    trv = pd.concat([tr, va])
    models = {k: (build_pipeline(k, feats, params).fit(trv[feats], trv["sales"]), feats) for k in ("ridge_log", "hgb")}
    f2 = feats + ["trend_index"]
    models["hgb+trend_index"] = (build_pipeline("hgb", f2, params).fit(trv[f2], trv["sales"]), f2)
    return models


def main() -> None:
    params, feats = load_params(), load_selected()
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    df = training_frame(raw)
    models = fit_models(df, params, feats)
    batches = {n: load_batch(n, raw) for n in ("normal", "drifted", "festival")}
    pd.set_option("display.width", 160)
    out: dict = {}

    def preds(b: pd.DataFrame) -> dict:
        p = {n: b[c].values for n, c in BASELINES.items()}
        p.update({k: score(m, b, f) for k, (m, f) in models.items()})
        return p

    print("A. WAPE (bias %) of every model on the production batches - models trained on data up to 2024-09-30")
    tab = {}
    for n, b in batches.items():
        tab[n] = {k: f"{M.wape(b.sales, v):.3f} ({M.bias_pct(b.sales, v):+.1%})" for k, v in preds(b).items()}
        out.setdefault("A", {})[n] = {k: dict(wape=round(M.wape(b.sales, v), 4), bias_pct=round(M.bias_pct(b.sales, v), 4)) for k, v in preds(b).items()}
    print(pd.DataFrame(tab).to_string())
    print("   normal  : business as usual -> models keep their validation accuracy (~0.13).")
    print("   festival: Diwali lead-up -> inputs drift (see drift check) but accuracy holds: the shift is EXPECTED and the lags follow it.")
    print("   drifted : level jumps to 150..210 -> the tree under-forecasts the most; the log-linear model follows better;")
    print("             the naive baselines stay unbiased but noisy (they only copy the recent past).")

    print("\nB. DRIFTED batch by week: bias % (negative = under-forecast = lost sales)")
    b = batches["drifted"].copy()
    b["week"] = (b["date"] - b["date"].min()).dt.days // 7 + 1
    p = preds(b)
    wk = pd.DataFrame({k: pd.Series(v, index=b.index).groupby(b.week).sum() for k, v in p.items()})
    act = b.groupby("week")["sales"].sum()
    wb = wk.sub(act, axis=0).div(act, axis=0)
    wb = wb[wb.index <= 8]
    print((wb[["hgb", "ridge_log", "seasonal_naive", "moving_average_7"]] * 100).round(1).T.to_string())
    out["B"] = (wb * 100).round(1).to_dict()

    print("\nC. STRESS TEST - what if the business were k times bigger? (sales AND lag features x k, normal batch)")
    nb = batches["normal"]
    rows = []
    for k in (1.0, 1.25, 1.5, 2.0, 3.0):
        s = nb.copy()
        s["sales"] = s["sales"] * k
        s[LEVEL_COLS] = s[LEVEL_COLS] * k
        row = dict(k=k)
        for name in ("hgb", "ridge_log"):
            m, f = models[name]
            pr = score(m, s, f)
            row[f"{name}_bias%"] = round(100 * M.bias_pct(s.sales, pr), 1)
            row[f"{name}_wape"] = round(M.wape(s.sales, pr), 3)
        row["hgb_max_forecast"] = round(float(score(models["hgb"][0], s, models["hgb"][1]).max()))
        rows.append(row)
    st = pd.DataFrame(rows)
    print(st.to_string(index=False))
    p99 = float(df["sales"].quantile(0.99))
    print(f"   99% of training sales are below {p99:.0f} units.  Whatever k is, the largest tree forecast stays near its CEILING")
    print("   (the 'hgb_max_forecast' column barely moves) - that is the leaf-average limit.  The log-linear model keeps rising with")
    print("   the inputs, so its bias grows about half as fast; it still under-forecasts because Ridge shrinks its weights toward")
    print("   the training average.  Neither is perfect: the real cure for a new level is retraining (and scale-free targets).")
    out["C"] = st.to_dict("records")

    print("\nD. ABLATION - give the tree a `trend_index` clock (days since 2023-01-01)")
    for n in ("drifted", "normal"):
        a, c = out["A"][n]["hgb"], out["A"][n]["hgb+trend_index"]
        print(f"   {n:<8} hgb WAPE {a['wape']:.3f} bias {a['bias_pct']:+.1%}   |   hgb+trend_index WAPE {c['wape']:.3f} bias {c['bias_pct']:+.1%}")
    print("   The clock lets the tree say 'late dates = newer, higher level' - but it can only repeat the LAST level it saw.")
    print("   It would fail again after the next jump; the principled fix is retraining (and scale-free targets).")

    # ------------------------------------------------------------------ figure (colors: reference palette, + line styles)
    ink, c_hgb, c_ridge, c_naive = "#0b0b0b", "#eb6834", "#2a78d6", "#1baf7a"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#fcfcfb")
    daily = b.assign(hgb=p["hgb"], ridge=p["ridge_log"], snaive=p["seasonal_naive"]).groupby("date")[["sales", "hgb", "ridge", "snaive"]].mean()
    r = daily.rolling(7, min_periods=3).mean()
    ax[0].plot(r.index, r.sales, color=ink, lw=2, label="actual")
    ax[0].plot(r.index, r.hgb, color=c_hgb, lw=2, ls="-", label="gradient boosting (tree)")
    ax[0].plot(r.index, r.ridge, color=c_ridge, lw=2, ls="--", label="Ridge on log lags")
    ax[0].plot(r.index, r.snaive, color=c_naive, lw=2, ls=":", label="seasonal naive")
    ax[0].set_title("DRIFTED production: average units per series-day (7-day mean)", fontsize=10, color=ink, loc="left")
    ax[0].set_ylabel("units per series-day")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].plot(st.k, st["hgb_bias%"], color=c_hgb, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label="gradient boosting (tree)")
    ax[1].plot(st.k, st["ridge_log_bias%"], color=c_ridge, lw=2, ls="--", marker="s", ms=6, mec="#fcfcfb", mew=2, label="Ridge on log lags")
    ax[1].axhline(0, color="#52514e", lw=0.8)
    ax[1].set_title("Stress test: bias % when the business is k times bigger", fontsize=10, color=ink, loc="left")
    ax[1].set_xlabel("k  (1 = training-like level)")
    ax[1].set_ylabel("bias % (negative = under-forecast)")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    for a_ in ax:
        a_.set_facecolor("#fcfcfb")
        a_.grid(color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(colors="#52514e", labelsize=8)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "trend_shift_extrapolation.png", dpi=130)
    print(f"\nfigure saved: reports/trend_shift_extrapolation.png")
    save_fact("trend_shift", out)


if __name__ == "__main__":
    main()
