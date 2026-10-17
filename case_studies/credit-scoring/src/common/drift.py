"""Drift detection with the two-sample Kolmogorov-Smirnov (KS) test.

Reference distribution  = what the model saw in training.
Production distribution = what the model sees now.

H0 (null hypothesis): both samples come from the same distribution.
A small p-value is evidence AGAINST H0 -> the distributions differ.
A large p-value does NOT prove "no drift"; it only means we lack evidence of a difference.

Because p-values shrink as samples grow, we ALSO look at the KS statistic D (the effect size,
the largest vertical gap between the two empirical CDFs) and apply a practical threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift_report(reference: pd.DataFrame, production: pd.DataFrame, features: list[str],
                    alpha: float = 0.05, min_effect: float = 0.10, min_rows: int = 100) -> pd.DataFrame:
    """Return one row per feature: KS statistic, p-value, decision and a plain-English verdict.

    statistically_significant : p < alpha            (evidence the distributions differ)
    practically_significant   : D >= min_effect      (the difference is big enough to care about)
    status                    : 'insufficient_data' | 'no_evidence_of_drift' | 'significant_but_small' | 'drift'
    """
    rows = []
    for f in features:
        ref = pd.to_numeric(reference[f], errors="coerce").dropna().to_numpy()
        prod = pd.to_numeric(production[f], errors="coerce").dropna().to_numpy()
        if len(prod) < min_rows or len(ref) < min_rows:
            rows.append(dict(feature=f, n_ref=len(ref), n_prod=len(prod), ks_statistic=np.nan, p_value=np.nan,
                             statistically_significant=False, practically_significant=False,
                             status="insufficient_data"))
            continue
        res = ks_2samp(ref, prod)
        sig = bool(res.pvalue < alpha)
        prac = bool(res.statistic >= min_effect)
        if sig and prac:
            status = "drift"
        elif sig:
            status = "significant_but_small"
        else:
            status = "no_evidence_of_drift"
        rows.append(dict(feature=f, n_ref=len(ref), n_prod=len(prod), ks_statistic=round(float(res.statistic), 4),
                         p_value=float(res.pvalue), statistically_significant=sig,
                         practically_significant=prac, status=status))
    return pd.DataFrame(rows)


def explain_row(r: pd.Series, alpha: float = 0.05) -> str:
    """Human readable explanation used in reports and in class."""
    if r.status == "insufficient_data":
        return f"{r.feature}: too few rows (ref={r.n_ref}, prod={r.n_prod}); no decision."
    verdict = {
        "drift": "reject H0: distributions differ AND the gap is large enough to matter.",
        "significant_but_small": "reject H0, but the gap (D) is small: statistically real, practically minor.",
        "no_evidence_of_drift": "fail to reject H0: no evidence of a difference (this is NOT proof of no drift).",
    }[r.status]
    return f"{r.feature}: D={r.ks_statistic:.3f}, p={r.p_value:.2g} (alpha={alpha}) -> {verdict}"
