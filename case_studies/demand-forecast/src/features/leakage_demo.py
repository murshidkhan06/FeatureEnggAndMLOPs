"""Leakage and split lessons.  Run:  python -m src.features.leakage_demo

The question to ask about EVERY feature and EVERY split:
    "Would this information really be available at the moment we make the prediction?"

We forecast at the end of day D for day D+1, so only data up to day D exists.

LESSON 1 - random split vs chronological split
  A shuffled train_test_split puts days from BEFORE and AFTER every test day into training.
  Two things go wrong at once:
    (a) the model sees the future of the test day (level, promotions, weather of the days around it), and
    (b) because lag_1 of day t+1 IS the sales of day t, the TEST LABEL sits inside a TRAINING FEATURE.
  We measure it with a controlled experiment on the SAME test rows:
     time-split model  : trained on days before the test period only (what production will look like)
     random-split model: trained on everything except the sampled test rows (past AND future neighbours)

LESSON 2 - a centred rolling window  (rolling(7, center=True))  includes 3 FUTURE days and today.
LESSON 3 - fitting the scaler on all data leaks the future level into the preprocessing.
          (Tiny effect on error here, visible in the min-max range: that is why every step lives in a Pipeline.)
LESSON 4 - actual temperature of the forecast day is not known at prediction time (we only have a FORECAST).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import DATA_RAW
from src.features.build_features import training_frame
from src.features.select_features import load_selected
from src.models import metrics as M
from src.models.pipelines import build_pipeline


def leaky_features(raw: pd.DataFrame) -> pd.DataFrame:
    """WRONG ways to build rolling features (kept ONLY to show the damage).  Returns the correct frame + leaky columns."""
    f = training_frame(raw)
    g = f.groupby(["product_id", "store_id"], sort=False)["sales"]
    f["rolling_mean_7_centered"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=4).mean())   # t-3 .. t+3
    f["rolling_mean_7_noshift"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())                 # t-6 .. t
    return f


def light(params: dict) -> dict:
    """A smaller boosting model (100 trees) so the demo runs in seconds; every number below uses the SAME light model."""
    return {**params, "hgb": {**params["hgb"], "max_iter": 100, "learning_rate": 0.1}}


def fit_scores(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], params: dict, kind: str = "hgb") -> dict:
    """Fit on `train`, score on `test`: returns WAPE and MAE (both: the lesson is the same for either metric)."""
    pipe = build_pipeline(kind, feats, params).fit(train[feats], train["sales"])
    f = np.clip(pipe.predict(test[feats]), 0, None)
    return dict(wape=M.wape(test["sales"], f), mae=M.mae(test["sales"], f))


def fit_score(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], params: dict, kind: str = "hgb", metric=M.wape) -> float:
    pipe = build_pipeline(kind, feats, params).fit(train[feats], train["sales"])
    return metric(test["sales"], np.clip(pipe.predict(test[feats]), 0, None))


def label_in_training_features(df: pd.DataFrame, seed: int = 0) -> float:
    """Share of TEST labels that already appear (as lag_1, lag_7 or lag_14) inside some TRAINING row after a random 80/20 split.

    Worked example: test row = (P01,S1, Mon 10 Mar, sales 120).  The training row (P01,S1, Tue 11 Mar) has lag_1 = 120.
    The model never sees 120 as a target, but it sits in a training FEATURE - the answer key is in the textbook."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    d = pd.to_datetime(df["date"])
    key = lambda frame, lag: set(zip(frame["product_id"], frame["store_id"], pd.to_datetime(frame["date"]) - pd.Timedelta(days=lag)))
    present = key(train, 1) | key(train, 7) | key(train, 14)
    hit = [(p, s, t) in present for p, s, t in zip(test["product_id"], test["store_id"], pd.to_datetime(test["date"]))]
    del d
    return float(np.mean(hit))


def lesson_random_vs_time(df: pd.DataFrame, feats, params, seeds=(0, 1, 2)) -> dict:
    split = params["split"]
    d = df["date"]
    test_mask = (d > pd.Timestamp(split["valid_end"])) & (d <= pd.Timestamp(split["test_end"]))
    past_mask = d <= pd.Timestamp(split["valid_end"])
    test_all = df[test_mask]
    res = {k: dict(wape=[], mae=[]) for k in ("time", "random_same_rows", "random_classic")}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        # sample 20% of the TEST-period days (whole days, all series) = the rows we will score in both set-ups
        days = test_all["date"].drop_duplicates()
        pick = set(rng.choice(days.values, size=int(0.2 * len(days)), replace=False))
        sampled = test_all[test_all["date"].isin(pick)]
        rest = df[~df["date"].isin(pick) & (df["date"] <= pd.Timestamp(split["test_end"]))]      # past AND future neighbours
        idx = rng.permutation(len(df))                                                           # classic student mistake: shuffle ALL rows 80/20
        cut = int(0.8 * len(df))
        for key, (tr_, te_) in dict(time=(df[past_mask], sampled), random_same_rows=(rest, sampled),
                                    random_classic=(df.iloc[idx[:cut]], df.iloc[idx[cut:]])).items():
            sc = fit_scores(tr_, te_, feats, params)
            res[key]["wape"].append(sc["wape"]); res[key]["mae"].append(sc["mae"])
    return {f"{k}_{m}": float(np.mean(v[m])) for k, v in res.items() for m in ("wape", "mae")}


def main() -> None:
    params, feats = light(load_params()), load_selected()
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    f = leaky_features(raw)
    split = params["split"]
    tr = f[f["date"] <= pd.Timestamp(split["valid_end"])]
    te = f[(f["date"] > pd.Timestamp(split["valid_end"])) & (f["date"] <= pd.Timestamp(split["test_end"]))]
    out: dict = {}

    print("LESSON 1 - RANDOM split vs CHRONOLOGICAL split  (light HistGradientBoosting, mean of 3 seeds, lower = better)")
    r = lesson_random_vs_time(f, feats, params)
    share = label_in_training_features(f)
    print(f"  {'set-up':<58} {'WAPE':>7} {'MAE (units)':>12}")
    print(f"  {'WRONG  shuffled 80/20 over all rows (other test rows!)':<58} {r['random_classic_wape']:7.4f} {r['random_classic_mae']:12.2f}")
    print(f"  {'WRONG  same test days, future neighbours also in train':<58} {r['random_same_rows_wape']:7.4f} {r['random_same_rows_mae']:12.2f}")
    print(f"  {'RIGHT  chronological: train on the past only':<58} {r['time_wape']:7.4f} {r['time_mae']:12.2f}")
    gap = (r["time_wape"] - r["random_same_rows_wape"]) / r["time_wape"]
    gap_mae = (r["time_mae"] - r["random_same_rows_mae"]) / r["time_mae"]
    print(f"  Controlled comparison on IDENTICAL rows (rows 2 vs 3): the random split understates the error by {gap:.1%} (WAPE) / {gap_mae:.1%} (MAE).")
    print("  (Row 1 scores different rows, so it cannot be compared fairly with row 3 - one more reason to control the experiment.)")
    print(f"  Direct proof of the leak: after a random 80/20 split, {share:.0%} of the TEST labels already appear as lag_1/lag_7/lag_14")
    print("  inside TRAINING rows (the answer is printed in the textbook).  After a chronological split this share is 0%.")
    print("  HONEST NOTE: the optimism is SMALL here because one-day-ahead lag features already carry most of the signal and the")
    print("  boosting model cannot memorise single days.  Do not conclude 'random split is fine': (1) a few percent of error is exactly the")
    print("  size of a real model improvement, (2) lessons 2-4 below show leaks that are LARGE, (3) the principle - never train on the future -")
    print("  does not depend on the size of the gap.")
    out["lesson1"] = dict(r, optimism_pct=round(gap * 100, 1), optimism_pct_mae=round(gap_mae * 100, 1), label_in_train_features=round(share, 3))

    print("\nLESSON 2 - centred rolling window leaks the answer")
    for name, cols in (("correct features (shift(1) before rolling)", feats),
                       ("+ rolling_mean_7_noshift  (window t-6..t, includes TODAY)", feats + ["rolling_mean_7_noshift"]),
                       ("+ rolling_mean_7_centered (window t-3..t+3, includes the FUTURE)", feats + ["rolling_mean_7_centered"])):
        w = fit_score(tr, te, cols, params)
        print(f"  {name:<68} test WAPE = {w:.4f}")
        out[f"lesson2_{cols[-1] if len(cols) > len(feats) else 'correct'}"] = round(w, 4)
    print("  Why: for day t the centred window = mean(sales t-3 .. t+3) contains sales(t) itself - the target.")
    print("  In production, day t+1..t+3 do not exist yet, so the feature cannot be computed: the model is unusable.")

    print("\nLESSON 3 - scaler fitted on ALL data (including the future) vs on TRAIN only")
    lvl = "rolling_mean_7"
    wrong = MinMaxScaler().fit(f[[lvl]])
    right = MinMaxScaler().fit(tr[[lvl]])
    top_wrong, top_right = wrong.transform(te[[lvl]]).max(), right.transform(te[[lvl]]).max()
    print(f"  test-period max of scaled {lvl}:  scaler fit on ALL data -> {top_wrong:.3f}   |  scaler fit on TRAIN only -> {top_right:.3f}")
    print(f"  train-only scaling shows values above 1.0 ({top_right:.2f}): the test level exceeds anything seen in training - an honest warning.")
    print("  The leaked scaler squeezes the future into [0, 1] and HIDES that shift.  Mean used for centering would also")
    print(f"  differ: all data {f[lvl].mean():.1f} vs train only {tr[lvl].mean():.1f} units (future level leaking into preprocessing).")
    out["lesson3"] = dict(max_scaled_all=round(float(top_wrong), 3), max_scaled_train=round(float(top_right), 3),
                          mean_all=round(float(f[lvl].mean()), 1), mean_train=round(float(tr[lvl].mean()), 1))

    print("\nLESSON 4 - ACTUAL temperature instead of the FORECAST")
    base = fit_score(tr, te, feats, params)
    sw = [c for c in feats if c != "forecast_temp"] + ["temperature"]
    leaky = fit_score(tr, te, sw, params)
    print(f"  with forecast_temp (available at prediction time)      test WAPE = {base:.4f}")
    print(f"  with actual temperature (known only AFTER the day)      test WAPE = {leaky:.4f}   <- slightly better, but unusable")
    print("  The gain is small (weather is a weak driver here) but the principle matters: the registry marks `temperature` as Rejected.")
    out["lesson4"] = dict(forecast_temp=round(base, 4), actual_temp=round(leaky, 4))

    print("\nRULE OF THUMB: every feature must have an entry 'available at prediction time = yes' in configs/feature_registry.yaml.")
    save_fact("leakage_demo", out)


if __name__ == "__main__":
    main()
