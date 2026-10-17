"""Baselines: models with NO learning.  Every ML model must beat these, otherwise it is not worth its complexity.

  naive                  forecast(t) = sales(t-1)                  "tomorrow = today"       (uses lag_1)
  seasonal_naive         forecast(t) = sales(t-7)                  "tomorrow = same weekday last week"  (lag_7)
  moving_average_7       forecast(t) = mean(sales t-7 .. t-1)      "tomorrow = recent average"  (rolling_mean_7)

Worked example: sales Mon..Sun last week = [80, 78, 82, 90, 120, 150, 140].  Forecast for next Saturday:
  naive = 120 (Friday), seasonal_naive = 150 (last Saturday), moving_average_7 = 105.7.
Seasonal naive is hard to beat when the weekly pattern is strong - and it keeps working when the level shifts,
because it always looks at the recent past.
"""
from __future__ import annotations

import pandas as pd

BASELINES = {"naive": "lag_1", "seasonal_naive": "lag_7", "moving_average_7": "rolling_mean_7"}


def baseline_forecast(name: str, frame: pd.DataFrame) -> pd.Series:
    return frame[BASELINES[name]].astype(float)
