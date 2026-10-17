"""The HISTORY STORE: a small in-memory stand-in for an ONLINE FEATURE STORE.

WHY it exists
-------------
A real-time request only says "forecast product P04, store S2, date 2025-03-01, price 56, no promotion".
It does NOT contain yesterday's sales - the caller should not have to know that lag_1 and rolling_mean_30 exist.
The service therefore keeps the recent sales history of every series and builds the lag/rolling features itself,
with the SAME `compute_features` function training used.  (Offline store = the training CSV / features table,
online store = this object.  Same code on both sides = no training-serving skew.)

In production the history would live in Redis / a feature-store service and be refreshed by a nightly job.
Here it is loaded from `data/processed/sales_history.csv` (override with the HISTORY_PATH environment variable).

Point-in-time correctness: for a request with forecast_date = t the store only uses rows with date < t,
even if the file also contains later days (so replaying old dates for testing never leaks the future).

Freshness: if the newest history row is older than forecast_date - 1 the history is STALE.
  staleness_days = (forecast_date - last_history_date) - 1        0 = perfectly fresh (history ends yesterday)
Up to `max_stale_days` (3) the store still answers (with a warning); beyond that it refuses.  Why 3?  rolling_mean_7 needs
at least 4 real days inside its 7-day window (min_periods=4), so at most 3 days may be missing.  A forecast built
from two-week-old lags is worse than no forecast.   Tiny example: history ends 2025-02-28, request for
2025-03-01 -> staleness 0;  request for 2025-03-04 -> staleness 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import REQUIRED, check_history, compute_features
from src.features.definitions import KEYS

TAIL_DAYS = 60            # enough for lag_14 and rolling_mean_30 (+ safety)
MIN_HISTORY_DAYS = 30


class HistoryStore:
    def __init__(self, history: pd.DataFrame, tail_days: int = TAIL_DAYS, max_stale_days: int = 3):
        check_history(history)                       # duplicates => refuse to start (fail loudly)
        h = history[REQUIRED].copy()
        h["date"] = pd.to_datetime(h["date"])
        self.tail_days, self.max_stale_days = tail_days, max_stale_days
        self._series = {k: g.sort_values("date").reset_index(drop=True) for k, g in h.groupby(KEYS)}
        self.last_date = h["date"].max()

    @classmethod
    def from_csv(cls, path, **kw) -> "HistoryStore":
        return cls(pd.read_csv(path, parse_dates=["date"]), **kw)

    # ------------------------------------------------------------------ info
    def series_ids(self) -> list[tuple[str, str]]:
        return sorted(self._series)

    def products(self) -> list[str]:
        return sorted({p for p, _ in self._series})

    def stores(self) -> list[str]:
        return sorted({s for _, s in self._series})

    def last_date_for(self, product_id: str, store_id: str) -> pd.Timestamp:
        return self._series[(product_id, store_id)]["date"].max()

    def staleness_days(self, product_id: str, store_id: str, forecast_date) -> int:
        """Days of missing history just before forecast_date (0 = fresh)."""
        fd = pd.Timestamp(forecast_date)
        last = self.last_date_for(product_id, store_id)
        return int(max(0, (fd - last).days - 1))

    # ------------------------------------------------------------------ features
    def features_for(self, product_id: str, store_id: str, forecast_date, price: float, discount: float = 0.0,
                     promotion: int = 0, holiday: int = 0, forecast_temp: float | None = None) -> pd.DataFrame:
        """One-row DataFrame with every feature for (series, forecast_date).  Raises ValueError (-> HTTP 422)."""
        key = (product_id, store_id)
        if key not in self._series:
            raise ValueError(f"unknown series product_id={product_id!r}, store_id={store_id!r}; "
                             f"known products {self.products()}, stores {self.stores()}")
        fd = pd.Timestamp(forecast_date)
        stale = self.staleness_days(product_id, store_id, fd)
        if stale > self.max_stale_days:
            raise ValueError(f"history store is {stale} days behind forecast_date {fd.date()} "
                             f"(last history day {self.last_date_for(*key).date()}, limit {self.max_stale_days}); "
                             "refresh the history before forecasting")
        h = self._series[key]
        window = h[h["date"] < fd].tail(self.tail_days)
        if len(window) < MIN_HISTORY_DAYS or (fd - window["date"].min()).days < MIN_HISTORY_DAYS:
            raise ValueError(f"not enough history for {key} before {fd.date()} (need >= {MIN_HISTORY_DAYS} days)")
        new = pd.DataFrame([dict(date=fd, product_id=product_id, store_id=store_id, sales=np.nan, price=price,
                                 discount=discount, promotion=promotion, holiday=holiday,
                                 forecast_temp=np.nan if forecast_temp is None else forecast_temp)])
        feats = compute_features(pd.concat([window, new], ignore_index=True))
        row = feats[feats["date"] == fd].iloc[[0]].reset_index(drop=True)
        if row[["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]].isna().any(axis=None):
            raise ValueError(f"history for {key} is too sparse to build lag features before {fd.date()}")
        return row

    def recent_frame(self, as_of=None, days: int = TAIL_DAYS) -> pd.DataFrame:
        """All series, last `days` rows (before `as_of`) - used by the batch forecaster."""
        parts = []
        for h in self._series.values():
            hh = h if as_of is None else h[h["date"] < pd.Timestamp(as_of)]
            parts.append(hh.tail(days))
        return pd.concat(parts, ignore_index=True)
