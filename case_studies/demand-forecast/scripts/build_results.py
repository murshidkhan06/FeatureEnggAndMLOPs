"""Merge reports/facts/*.json (written by every demo) + registry + captured pytest summary into results.json
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


reg = yaml.safe_load(open(CONFIGS / "feature_registry.yaml", encoding="utf-8"))["features"]
man = fact("feature_manual_examples", {})
prof = fact("data_profile", {})
train = fact("train", {})
ev = fact("evaluation", {})
pytest_txt = (ROOT / "docs" / "sample_outputs" / "08_pytest.txt")
summary = re.findall(r"=+ (.*(?:passed|failed).*) =+", pytest_txt.read_text()) if pytest_txt.exists() else []

formula = {
    "lag_1": ("sales.shift(1) per (product, store)", "sales of days 1..8 = 10,12,11,13,20,22,21,14 -> day 8", man.get("lags", {}).get("lag_1")),
    "lag_7": ("sales.shift(7)", "same series, day 8", man.get("lags", {}).get("lag_7")),
    "rolling_mean_7": ("sales.shift(1).rolling(7).mean()", "same series, day 8: mean(10,12,11,13,20,22,21)", man.get("rolling_mean_7", {}).get("code")),
    "rolling_mean_7 (WRONG, no shift)": ("sales.rolling(7).mean()", "same series, day 8: mean(12,11,13,20,22,21,14)", man.get("rolling_mean_7", {}).get("wrong_no_shift")),
    "price_change": ("price / price_yesterday - 1", "price 40 -> 38", man.get("price", {}).get("price_change")),
    "discount_pct": ("discount / (price + discount)", "price 38, discount 2", man.get("price", {}).get("discount_pct")),
    "dow_sin/dow_cos": ("sin/cos(2*pi*day_of_week/7)", "Monday vs Sunday distance on the circle", man.get("cyclical")),
}
feature_table = [dict(name=k, formula=v[0], example_input=v[1], example_output=v[2]) for k, v in formula.items()]

retrain = facts_matching("retrain_")
results = dict(
    project="demand-forecast",
    dataset=dict(train_shape=prof.get("train_shape"), series=prof.get("series"), date_range=prof.get("date_range"),
                 production_rows=prof.get("production_rows"), target_summary=prof.get("target_rate"), missing_rates=prof.get("missing_rates"),
                 top_correlations_with_target=prof.get("top_correlations_with_target"), top_feature_pairs=prof.get("top_feature_pairs"),
                 weekly_average_table=prof.get("weekly_average_table"), zero_rows=prof.get("zero_rows"), outlier_candidates=prof.get("outlier_candidates")),
    feature_table=feature_table,
    feature_registry=[dict(name=r["name"], role=r["role"], status=r["status"], availability=r["prediction_time_availability"]) for r in reg],
    feature_selection=fact("selection"),
    leakage_demo=fact("leakage_demo"),
    models=dict(champion_rule="lowest validation WAPE among candidates with |bias| <= 3%", chosen=train.get("best"), final_test_metrics=train.get("final_test"),
                baselines_valid=train.get("baselines_valid"), baselines_test=train.get("baselines_test"),
                candidates_valid=train.get("candidates_valid"), interval=train.get("interval"), rows=train.get("rows")),
    evaluation=ev,
    trend_shift=fact("trend_shift"),
    explainability=fact("explain"),
    drift=dict(normal=fact("drift_batch_normal"), normal_seasonal_reference=fact("drift_batch_normal_seasonal"),
               drifted=fact("drift_batch_drifted"), festival=fact("drift_batch_festival"),
               festival_seasonal_reference=fact("drift_batch_festival_seasonal"),
               performance=dict(normal=fact("performance_batch_normal"), drifted=fact("performance_batch_drifted"))),
    retraining=retrain,
    rolling_origin_optional=fact("rolling_origin"),
    batch_forecast=fact("batch_forecast"), batch_backtest=fact("batch_backtest"),
    pytest_summary=summary[-1] if summary else None,
    api_latency_ms=fact("api_latency"),
    components=dict(
        CORE=["chronological time split", "lag/rolling features with shift(1)", "baseline comparison (naive, seasonal naive, moving average)",
              "Ridge-on-log and gradient boosting in sklearn Pipelines", "WAPE / bias / MAE", "feature registry + approval gate",
              "KS drift + decision ladder", "scheduled (monthly) + trigger-based retraining", "champion/challenger on the same recent hold-out",
              "batch forecast (recursive, 7 days)", "FastAPI real-time endpoint + history store", "MLflow, DVC, Docker, Kubernetes, Prometheus/Grafana, CI"],
        OPTIONAL=["TimeSeriesSplit rolling-origin evaluation", "quantile-style prediction intervals (relative residual quantiles)", "SHAP", "cyclical sin/cos encodings"]),
    failure_scenarios_reproduced=["(a) stale history store -> degraded forecasts, staleness warning, refusal beyond 3 days",
                                  "(b) MAPE infinite on zero-sales days -> use WAPE", "(c) duplicated dates after a pipeline retry -> data test + idempotent upsert",
                                  "(d) price unit change x100 -> data-quality price_unit check + API range check",
                                  "tree model cannot extrapolate on the level shift", "festival drift without model failure (expected seasonal shift)",
                                  "guardrail blocks a biased challenger even with human sign-off"],
    failure_scenarios=fact("failure_scenarios"),
)
(ROOT / "results.json").write_text(json.dumps(results, indent=1, default=str))
print("results.json written:", len(json.dumps(results, default=str)) // 1024, "KB; sections:", list(results))
