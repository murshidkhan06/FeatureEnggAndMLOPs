"""Credit-risk metrics, each with the formula and a tiny example you can verify by hand.

y = 1 means "defaulted", p = model's probability of default (PD).

ROC-AUC      probability that a random defaulter gets a HIGHER PD than a random good customer.
             Example: scores good=[0.1,0.2], bad=[0.15,0.9] -> pairs: (0.15>0.1)(0.15<0.2)(0.9>0.1)(0.9>0.2) -> 3/4 = 0.75
Gini         = 2*AUC - 1  (the credit industry's rescaling: 0 = random, 1 = perfect).  AUC 0.75 -> Gini 0.50
KS statistic the biggest gap between the CDF of scores of defaulters and of good customers.  It is exactly the
             two-sample Kolmogorov-Smirnov statistic D that we also use for DRIFT, applied to two groups of
             customers instead of two time periods.  0 = no separation, 1 = perfect separation.
PR-AUC       area under the precision-recall curve (more informative than ROC when defaults are rare, ~10%).
Recall @ approval rate  "if we approve the safest 80% and decline the riskiest 20%, what share of all defaulters
             did we decline?"  Example: 100 defaulters, 40 of them in the declined 20% -> 0.40.
Calibration  among customers with predicted PD around 10%, do ~10% really default?  (reliability table)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ks_statistic(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def recall_at_approval_rate(y, p, approval_rate: float = 0.80) -> float:
    y, p = np.asarray(y), np.asarray(p)
    cutoff = np.quantile(p, approval_rate)               # approve everyone below this PD
    declined = p > cutoff
    return float((declined & (y == 1)).sum() / max((y == 1).sum(), 1))


def credit_metrics(y, p, approval_rate: float = 0.80) -> dict:
    auc = roc_auc_score(y, p)
    return dict(roc_auc=round(float(auc), 4), gini=round(float(2 * auc - 1), 4), ks_stat=round(ks_statistic(y, p), 4),
                pr_auc=round(float(average_precision_score(y, p)), 4),
                recall_at_80_approval=round(recall_at_approval_rate(y, p, approval_rate), 4),
                brier=round(float(brier_score_loss(y, p)), 4), n=int(len(y)), default_rate=round(float(np.mean(y)), 4))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Sort by predicted PD, cut into equal-size groups, compare average prediction with the observed default rate."""
    d = pd.DataFrame(dict(y=np.asarray(y), p=np.asarray(p)))
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    t = d.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted_pd=("p", "mean"), observed_default_rate=("y", "mean"))
    return t.round(4).reset_index()


def business_view(loan_amount, y, p, approval_rate: float = 0.80, lgd: float = 0.60, margin: float = 0.08) -> dict:
    """Money view with STATED assumptions (change them in configs/params.yaml).

    We decline the riskiest (1 - approval_rate) share of applicants.
      loss avoided   = loan_amount x LGD            for declined applicants who WOULD have defaulted
      margin lost    = loan_amount x margin         for declined applicants who would have repaid (good customers rejected)
      net benefit    = loss avoided - margin lost

    Tiny example: declined defaulter with loan 100,000 -> avoided loss 60,000;
                  declined good customer with loan 100,000 -> lost margin 8,000.
    """
    la, y, p = np.asarray(loan_amount, dtype=float), np.asarray(y), np.asarray(p)
    declined = p > np.quantile(p, approval_rate)
    loss_avoided = float((la[declined & (y == 1)] * lgd).sum())
    margin_lost = float((la[declined & (y == 0)] * margin).sum())
    return dict(approval_rate=approval_rate, declined=int(declined.sum()),
                defaulters_declined=int((declined & (y == 1)).sum()), good_customers_declined=int((declined & (y == 0)).sum()),
                expected_loss_without_model=round(float((la[y == 1] * lgd).sum())), loss_avoided=round(loss_avoided),
                margin_lost=round(margin_lost), net_benefit=round(loss_avoided - margin_lost))
