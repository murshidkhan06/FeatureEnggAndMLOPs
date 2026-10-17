"""Forecast error metrics - and what each one means for a store manager.

Notation: y = actual units sold, f = forecast units, e = f - y (positive = we OVER-forecast).

Worked example  (3 days)    y = [100, 0, 50]     f = [90, 10, 60]     e = [-10, +10, +10]
  MAE   = mean|e|            = (10+10+10)/3 = 10 units     "on average we are 10 units off per day"
  RMSE  = sqrt(mean e^2)     = 10 units                    punishes big misses more than MAE
  WAPE  = sum|e| / sum y     = 30 / 150 = 0.20 (20%)       "20% of the units we sold were mis-forecast" - safe with zeros
  bias  = mean(e)            = +3.33 units                 > 0 : we systematically forecast too much (stock piles up)
  MAPE  = mean(|e| / y)      = (0.10 + inf + 0.20)/3 = inf  y = 0 on a stock-out day -> division by zero. DO NOT USE for
                                                           intermittent / stock-out data; use WAPE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(f) - np.asarray(y))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(f) - np.asarray(y)) ** 2)))


def wape(y, f) -> float:
    """Weighted absolute percentage error = total absolute error / total actual units."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(f - y).sum() / y.sum())


def bias(y, f) -> float:
    """Mean error in units.  Positive = over-forecast."""
    return float(np.mean(np.asarray(f) - np.asarray(y)))


def bias_pct(y, f) -> float:
    """Bias as a share of average sales:  sum(f - y) / sum(y)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / y.sum())


def mape_naive(y, f) -> float:
    """Classic MAPE (kept ONLY to show why it breaks).  Returns inf/nan when y contains zeros."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs(f - y) / y))


def high_demand_auc(y, f, level) -> float:
    """roc_auc of 'will tomorrow be a HIGH-demand day?' (y > 1.15 x recent level) scored by f / level.

    Why an AUC in a regression project?  It answers a planner's question - "can the forecast rank the
    spike days (promotions, holidays, weekends) above ordinary days?" - and keeps the quality-gate contract
    (every project's champion exposes `roc_auc`).  0.5 = no better than a coin flip."""
    y, f, level = (np.asarray(a, float) for a in (y, f, level))
    ok = np.isfinite(level) & (level > 0)
    label = (y[ok] > 1.15 * level[ok]).astype(int)
    if label.min() == label.max():
        return float("nan")
    return float(roc_auc_score(label, f[ok] / level[ok]))


def report(y, f, level=None) -> dict:
    """All metrics at once (rounded for logs)."""
    out = dict(mae=mae(y, f), rmse=rmse(y, f), wape=wape(y, f), bias=bias(y, f), bias_pct=bias_pct(y, f))
    if level is not None:
        out["roc_auc"] = high_demand_auc(y, f, level)
    return {k: round(v, 4) for k, v in out.items()}


def cost_report(y, f, over: float = 1.0, under: float = 3.0) -> dict:
    """Business view: over-forecast units become waste/holding cost, under-forecast units become lost sales."""
    e = np.asarray(f, float) - np.asarray(y, float)
    over_units, under_units = float(e[e > 0].sum()), float(-e[e < 0].sum())
    return dict(over_forecast_units=round(over_units), under_forecast_units=round(under_units),
                weighted_cost=round(over * over_units + under * under_units))


def weekly_wape(dates: pd.Series, y, f) -> pd.DataFrame:
    """WAPE and bias per calendar week (Mon-Sun)."""
    df = pd.DataFrame(dict(date=pd.to_datetime(dates).values, y=np.asarray(y, float), f=np.asarray(f, float)))
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    g = df.groupby("week")
    return pd.DataFrame(dict(rows=g.size(), actual=g.y.sum(), forecast=g.f.sum(),
                             wape=g.apply(lambda x: wape(x.y, x.f), include_groups=False),
                             bias_pct=g.apply(lambda x: bias_pct(x.y, x.f), include_groups=False))).reset_index()
