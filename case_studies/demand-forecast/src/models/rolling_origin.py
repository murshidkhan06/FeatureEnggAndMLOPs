"""OPTIONAL: rolling-origin evaluation with sklearn's TimeSeriesSplit.   python -m src.models.rolling_origin

WHY: one train/valid/test split gives ONE accuracy number from ONE period (which may be unusually easy or hard - e.g. a festival).
A rolling-origin evaluation repeats the exercise with several cut-off dates, always training on the PAST and testing on the NEXT block:

    fold 1:  train |=====|  test |--|
    fold 2:  train |========|     test |--|
    fold 3:  train |===========|  test |--|        (the training window EXPANDS; the test block moves forward in time)

TimeSeriesSplit works on row positions, so we split the sorted list of UNIQUE DATES (all 24 series of a day stay together) and map back to rows.
Worked example: 10 dates, 4 folds -> test blocks of 2 dates: fold 1 trains on dates 1-2 and tests 3-4, fold 2 trains on 1-4 and tests 5-6, ...
Not needed by any core step (OPTIONAL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import DATA_RAW
from src.features.build_features import training_frame
from src.features.leakage_demo import light
from src.features.select_features import load_selected
from src.models import metrics as M
from src.models.baselines import baseline_forecast
from src.models.pipelines import build_pipeline
from src.models.train import score


def rolling_origin(df: pd.DataFrame, feats: list[str], params: dict, n_splits: int = 5, test_days: int = 60) -> pd.DataFrame:
    dates = np.sort(df["date"].unique())
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_days)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(tss.split(dates), 1):
        tr = df[df["date"].isin(dates[tr_idx])]
        te = df[df["date"].isin(dates[te_idx])]
        assert tr["date"].max() < te["date"].min()                         # the past always precedes the test block
        r = dict(fold=k, train_to=str(tr["date"].max().date()), test_from=str(te["date"].min().date()), test_to=str(te["date"].max().date()))
        r["moving_average_7"] = M.wape(te["sales"], baseline_forecast("moving_average_7", te))
        for kind in ("ridge_log", "hgb"):
            pipe = build_pipeline(kind, feats, params).fit(tr[feats], tr["sales"])
            r[kind] = M.wape(te["sales"], score(pipe, te, feats))
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def main() -> None:
    params, feats = light(load_params()), load_selected()
    df = training_frame(pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"]))
    t = rolling_origin(df, feats, params)
    print("ROLLING-ORIGIN EVALUATION (OPTIONAL) - WAPE per fold, 5 folds x 60 test days, light boosting model")
    print(t.to_string(index=False))
    print(f"\nmean WAPE: moving_average_7 {t.moving_average_7.mean():.4f} | ridge_log {t.ridge_log.mean():.4f} | hgb {t.hgb.mean():.4f}")
    print(f"spread of the hgb WAPE across folds: {t.hgb.min():.3f} .. {t.hgb.max():.3f}  -> one split would have hidden this variation.")
    save_fact("rolling_origin", t.to_dict("records"))


if __name__ == "__main__":
    main()
