"""Collect the facts of one full run into results.json (machine-readable summary used by the course guide).

Run AFTER  python scripts/capture_outputs.py   (it needs reports/*.json and docs/sample_outputs/*.txt).

    python scripts/build_results.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import re

import pandas as pd
import yaml

from src import project as P
from src.common.drift import ks_drift_report
from src.common.paths import CONFIGS, DATA_PROCESSED, DATA_PRODUCTION, REPORTS
from src.features.build_features import engineer
from src.features.definitions import DEFINITIONS


def jload(name):
    f = REPORTS / name
    return json.loads(f.read_text()) if f.exists() else None


def drift_table(batch_file: str) -> dict:
    ref = pd.read_csv(DATA_PROCESSED / "reference.csv")
    prod = engineer(pd.read_csv(DATA_PRODUCTION / batch_file))
    rep = ks_drift_report(ref, prod, P.DRIFT_FEATURES)
    return dict(batch=batch_file, h0="reference and production values come from the same distribution", alpha=0.05, practical_threshold_D=0.10,
                rows=json.loads(rep.round(5).to_json(orient="records")))


def pytest_summary() -> dict:
    f = ROOT / "docs" / "sample_outputs" / "08_pytest.txt"
    if not f.exists():
        return {}
    txt = f.read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed(?:, (\d+) (?:skipped|failed))?.* in ([\d.]+)s", txt)
    return dict(passed=int(m.group(1)), other=m.group(2), seconds=float(m.group(3)), raw=m.group(0)) if m else dict(raw="see 08_pytest.txt")


def main() -> None:
    prof = jload("data_profile.json") or {}
    feat_ex = jload("feature_examples.json") or {}
    sel = pd.read_csv(REPORTS / "feature_selection.csv") if (REPORTS / "feature_selection.csv").exists() else pd.DataFrame()
    selected = yaml.safe_load((CONFIGS / "selected_features.yaml").read_text())
    ev, ex, lk, fr, age, met = (jload(n) for n in ("evaluation.json", "explain.json", "leakage_demo.json", "fairness_report.json", "age_experiment.json", "metrics.json"))
    retrain = jload("retrain_record.json")
    res = dict(
        project="credit-scoring",
        dataset=dict(shape=prof.get("shape"), target_rate=prof.get("target_rate"), missing_rates_pct=prof.get("missing_rates_pct"),
                     top_correlations=prof.get("top_correlations"), income=prof.get("income"), repeat_applications=prof.get("repeat_applications"),
                     income_missing_by_employment=prof.get("income_missing_by_employment"),
                     splits="time-based 70/15/15 by application_date, applicant-disjoint; see 00_make_data_and_split.txt"),
        feature_table=[dict(name=k, formula=v["formula"], definition=v["definition"], example_input=v["example_in"], example_output=v["example_out"])
                       for k, v in DEFINITIONS.items()],
        feature_hand_calculation=feat_ex,
        selection=dict(selected=selected["selected_features"], rejected=selected["rejected"],
                       table=json.loads(sel.to_json(orient="records")) if len(sel) else []),
        leakage_demo=lk,
        models=dict(comparison=met["comparison"] if met else None, champion_test=met["test"] if met else None,
                    feature_engineering_gain=dict(valid=met.get("feature_engineering_gain_valid_auc"), test=met.get("feature_engineering_gain_test_auc")) if met else None,
                    boosting_gain_valid_auc=met.get("boosting_gain_valid_auc") if met else None, decision="keep logistic-regression scorecard (boosting does not win by >= 0.02 AUC)"),
        evaluation=ev,
        explainability=dict(standardized_coefficients=ex["standardized_coefficients"][:10] if ex else None,
                            permutation_top=ex["permutation_importance"][:6] if ex else None,
                            shap_top=ex["shap_hgb_benchmark"][:6] if ex else None, reason_code_examples=ex["reason_code_examples"] if ex else None),
        fairness=fr, age_experiment=age,
        drift=dict(normal=drift_table("batch_normal.csv"), drifted=drift_table("batch_drifted.csv"),
                   last_check_report=jload("drift_report.json"),
                   significant_but_small_example="credit_history_years in batch_normal.csv: D about 0.05, p about 0.001 -> statistically significant, practically small -> NO_ACTION"),
        labelled_performance_drifted=(jload("drift_report.json") or {}).get("performance"),
        retraining=retrain,
        pytest=pytest_summary(),
        api_latency_ms=jload("api_latency.json"),
        core_components=["synthetic data generator", "feature engineering (engineer)", "filter/wrapper/embedded selection", "feature registry + approval guard",
                         "logistic-regression scorecard (sklearn Pipeline)", "MLflow tracking (local sqlite)", "metrics: AUC, KS, Gini, PR-AUC, recall@80% approval, calibration",
                         "reason codes", "FastAPI service + Prometheus metrics", "drift check (KS) + decision ladder", "champion/challenger + approval + rollback",
                         "fairness audit (pandas only)", "review gate + review pack", "pytest suite", "Dockerfile", "GitHub Actions CI", "DVC pipeline"],
        optional_components=["SHAP on the gradient-boosting benchmark", "HistGradientBoosting challenger (kept for monitoring, not shipped)",
                             "formal fairness libraries (not used)", "Kubernetes / Minikube manifests", "docker-compose Prometheus + Grafana stack",
                             "matplotlib figures (profile_data.py --plots)"],
        failure_scenarios_reproduced=["(a) approval rate jumps: income unit changed upstream -> data quality FAILED -> FIX_DATA_PIPELINE (16_failure_scenarios.txt)",
                                      "(b) AUC fell after a new feature: leaky column absent in production, offline 0.935 -> production ~0.25 (04_leakage_demo.txt, 16_failure_scenarios.txt)",
                                      "(c) API 422 spike after schema change: renamed field + capitalised category (16_failure_scenarios.txt)",
                                      "drift significant but small: credit_history_years, NO_ACTION (10_drift_normal.txt)",
                                      "downturn: drift + AUC drop 0.069 -> RETRAIN_CANDIDATE -> challenger -> approval -> promote -> rollback (11, 12)"],
        sample_outputs=sorted(p.name for p in (ROOT / "docs" / "sample_outputs").glob("*.txt")),
    )
    (ROOT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"results.json written ({(ROOT / 'results.json').stat().st_size // 1024} KB), sections: {list(res)}")


if __name__ == "__main__":
    main()
