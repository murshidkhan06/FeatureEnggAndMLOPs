"""Merge reports/facts/*.json (written by every demo) + the registry + the captured pytest summary into results.json
(machine-readable facts for the course guide).   python scripts/build_results.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re

import yaml

from src.common.paths import CONFIGS, REPORTS, ROOT

F = REPORTS / "facts"


def fact(name, default=None):
    p = F / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else default


def facts_matching(prefix):
    return {p.stem: json.loads(p.read_text()) for p in sorted(F.glob(f"{prefix}*.json"))}


def drift_view(d):
    """Compact view of one run_drift_check result."""
    if not d:
        return None
    return dict(batch=d["batch"], as_of=d["as_of"], today=d["today"], data_quality=d["data_quality"],
                ks_table=[dict(feature=r["feature"], D=r["ks_statistic"], p_value=r["p_value"], status=r["status"]) for r in d["drift"]],
                score_shift=d["score_shift"], pseudo_label=d["pseudo_label"],
                labelled=None if not d["labelled"] else {k: d["labelled"][k] for k in ("auc_drop", "recall_drop", "calibration_ratio", "mean_pd", "observed_rate", "contact_rate_ratio")} | {"metrics": d["labelled"]["metrics"]},
                decision=d["decision"]["decision"], reasons=d["decision"]["reasons"], H0="reference and production values come from the same distribution", alpha=0.05)


reg = yaml.safe_load(open(CONFIGS / "feature_registry.yaml", encoding="utf-8"))["features"]
man = fact("feature_manual_examples", {})
prof = fact("data_profile", {})
train = fact("train", {})
pytest_txt = ROOT / "docs" / "sample_outputs" / "08_pytest.txt"
summary = re.findall(r"=+ (.*(?:passed|failed).*) in [\d.]+s.*=+", pytest_txt.read_text()) if pytest_txt.exists() else []
m = man.get("features", {})
formula = [
    ("average_balance", "mean(balance_after) over transactions in (as_of - 90d, as_of]", "10 transactions since 2023-01-01 with balances 80,80,180,160,75,75,175,155,70,70 (thousand)"),
    ("balance_trend", "OLS slope of the 6 month-end balances (INR/month)", "month-end balances 95,90,85,80,75,70 (thousand); also [100, 80, 60] -> -20"),
    ("salary_credit_frequency", "salary credits in the 6-month window / 6", "5 salary credits in 6 months"),
    ("loan_payment_ratio", "loan payments made / loan payments due (6 months)", "5 paid, 6 due"),
    ("missed_payment_count", "sum of missed_payment over the last 6 PUBLISHED loan_status months", "Jan missed; the March row is published 2023-04-05 and must be ignored on 2023-03-31"),
    ("credit_utilization", "mean monthly card spend / credit_limit", "6,000 per month, limit 30,000"),
    ("recent_transaction_count", "transactions in the last 30 days", "4 transactions after 2023-03-01"),
    ("income_stability", "std / mean of salary amounts (sample std, n-1); NaN if fewer than 2 credits", "salaries 100,110,90,100,100 (thousand): std 7.07 / mean 100"),
    ("account_age", "whole months from signup_date to as_of", "signup 2020-03-15, as_of 2023-03-31"),
    ("util_x_missed", "credit_utilization * missed_payment_count", "0.2 * 1"),
    ("balance_trend_norm", "clip(balance_trend / (|average_balance| + 1), -1, 1)", "-5000 / 112001"),
    ("log_average_balance", "log1p(max(average_balance, 0))", "log1p(112000)"),
]
feature_table = [dict(name=n, formula=f, example_input=e, by_hand=m.get(n, {}).get("by_hand"), code_output=m.get(n, {}).get("code"), match=m.get(n, {}).get("match")) for n, f, e in formula]

drift = facts_matching("drift_")
retrain = facts_matching("retrain_")
lag = facts_matching("label_lag_")
results = dict(
    project="credit-risk-pd",
    business="Monthly early-warning model for EXISTING retail customers: P(90+ days past due within the next 6 months) on each month-end (as_of_date); collections contacts the riskiest 10 %.",
    dataset=dict(snapshot_shape=prof.get("shape"), raw_tables=prof.get("raw_tables"), target="default_next_6m", target_rate_train=prof.get("target_rate_train"),
                 split_table=prof.get("split_table"), missing_rates_pct=prof.get("missing_rates_pct"), income_stability_missing_rows_train=prof.get("income_stability_missing_rows_train"),
                 top_correlations_with_target=prof.get("top_correlations_with_target"), top_feature_pairs=prof.get("top_feature_pairs"),
                 outlier_credits=prof.get("outlier_credits"), segment_table=prof.get("segment_table"), defaults_per_year=prof.get("defaults_per_year"),
                 customers_ever_defaulting=prof.get("customers_defaulting")),
    feature_table=feature_table,
    hand_calculation_all_match=man.get("all_match"),
    feature_registry=[dict(name=r["name"], version=r["version"], status=r["status"], stage=r["stage"], owner=r["owner"], window=r["window"],
                           availability_lag_days=r["availability"]["lag_days"], prediction_time_availability=r["prediction_time_availability"],
                           rejection_reason=r.get("rejection_reason"), lifecycle=[f"{h['stage']} {h['date']} {h['by']}" for h in r["history"]]) for r in reg],
    feature_selection=fact("selection"),
    leakage_demo=fact("leakage_demo"),
    models=dict(rule="logistic regression is kept unless HistGradientBoosting beats it on VALIDATION by >= 0.005 ROC-AUC", chosen=train.get("winner"),
                val_gain=train.get("val_gain"), features=train.get("features"), rows=train.get("rows"), by_model=train.get("models"),
                final_test_metrics=train.get("final_test_metrics"), contact_threshold=train.get("contact_threshold")),
    evaluation=fact("evaluation"),
    explainability=fact("explain"),
    label_lag=lag,
    drift={k: drift_view(v) for k, v in drift.items()},
    retraining={k: {kk: vv for kk, vv in v.items() if kk != "drift_table"} for k, v in retrain.items()},
    batch_score=fact("batch_score"),
    pytest_summary=summary[-1] if summary else None,
    api_latency_ms=dict(fact("api_latency") or {}, note="client-side, 200 sequential POST /predict incl. HTTP, on a shared 2-core Linux VM that was busy with other jobs; an earlier run on an idle VM gave p50 ~17 ms / p95 ~27 ms. Treat as order-of-magnitude."),
    drift_exporter_traffic_demo=fact("traffic_demo"),
    components=dict(
        CORE=["prediction timestamp (as_of_date) + availability rule available_at <= as_of_date", "leakage-safe feature pipeline build_snapshot_features()",
              "assert_no_future_data() guard + future-perturbation test", "feature registry (metadata + lifecycle history) with a Production gate",
              "training refuses non-Approved/Production features", "time-based split with purge gap (never random)", "logistic regression baseline -> HistGradientBoosting candidate",
              "ROC-AUC, PR-AUC, recall@top-10%, lift, KS + business meaning", "permutation importance", "FastAPI /predict (feature vector in, PD out) + monthly batch scoring",
              "KS drift + data-quality gate + decision ladder with label-lag proxies", "champion/challenger retraining with human approval, guardrails and rollback",
              "MLflow, DVC, Docker, Kubernetes/Minikube, Prometheus/Grafana, GitHub Actions CI"],
        OPTIONAL=["SHAP (python -m src.models.explain --shap)", "gradient-boosting hyper-parameter search (not implemented; exercise)", "docker compose observability stack",
                  "Minikube deployment"]),
    failure_scenarios_reproduced=[
        "(a) leaky average_balance_leaky in training: test AUC ~0.96, live AUC far lower (04_leakage_demo, 13_failure_scenarios)",
        "(a2) post-default columns (days_past_due / recoveries / collections status of the target period) give AUC 0.94-1.0",
        "(a3) random split vs time split: optimistic AUC",
        "(b) salary_credit_frequency null rate jumps to ~40 % after a source migration -> FIX_DATA_PIPELINE, not retraining (11b, 13)",
        "(c) challenger trained on the stress period only: no gain on the stress hold-out and worse on calm data (13_failure_scenarios)",
        "label lag: labels for the drifted month do not exist for 6 months -> INVESTIGATE, then proxy evidence -> RETRAIN_CANDIDATE (11)",
        "significant-but-small drift (account_age) triggers nothing (10_drift_normal)",
        "promotion rules fail (ROC-AUC gain below the required bar) even with a human approver name (12)",
        "unpublished loan_status row would inflate missed_payment_count (02_feature_manual_examples)"],
)
(ROOT / "results.json").write_text(json.dumps(results, indent=1, default=str))
print("results.json written:", len(json.dumps(results, default=str)) // 1024, "KB; sections:", list(results))
