"""
src/alerts.py

THRESHOLD-BASED ALERTING: the last mile of monitoring. Drift checks, performance checks,
and API metrics are all useless if nobody (or nothing) actually looks at them regularly --
alerting is what turns "a number in a report file" into "something a human is told about."

WHY threshold-based, and why so simple? For a teaching project (and for a LOT of real
early-stage production systems), a small, explicit set of rules -- "if this number crosses
that line, say so clearly" -- is more auditable and more debuggable than something fancier
(e.g. anomaly-detection-on-the-metrics-themselves). You can look at any alert and know
EXACTLY why it fired. This module deliberately reuses the SAME thresholds the underlying
checks already use (params.yaml's drift / performance / alerting sections) rather than
inventing separate alert-only numbers, so there is exactly one place that defines
"what counts as a problem" per metric.

WHAT this module checks, each backed by a module already built in this project:
    1. Data drift        (numeric KS test + categorical chi-square, src/drift.py)
    2. Performance decay  (src/performance_monitor.py)
    3. API error rate     (src/monitoring.py's in-process Metrics, or a fresh /metrics call)

HOW alerts are delivered, in this local project: printed to stdout, written to
monitoring/app.log via the existing logger, AND collected into a single alert report file
(reports/alerts_report.json) -- exactly the "print / log / create an alert report" trio the
course spec asks for. See the module docstring's final paragraph for how a real production
system would extend this.

PRODUCTION EXTENSION (not implemented here, deliberately): a real system would replace or
supplement `_dispatch_alert()` with a call to an external alerting integration -- e.g. a
Slack webhook, PagerDuty/Opsgenie API, or an email via SES/SMTP. The trigger LOGIC above
(what counts as an alert, and why) would not need to change at all -- only where the alert
is SENT. Keeping the two concerns separate (decide vs. deliver) is exactly why
`_dispatch_alert()` is a single, small, easily-swappable function.

Run from the repo root:
    python -m src.alerts
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "reports" / "alerts_report.json"

import sys
sys.path.insert(0, str(REPO_ROOT))
from src.drift import check_drift, check_categorical_drift  # noqa: E402
from src.monitoring import logger, summarize_predictions_log  # noqa: E402
from src.performance_monitor import check_performance_decay  # noqa: E402


def load_params() -> dict:
    with open(REPO_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def _dispatch_alert(alert: Dict[str, Any]) -> None:
    """The single place an alert is actually delivered. For this local teaching project
    that's print + the shared logger; see the module docstring for how this would change
    in a real deployment (swap this one function, nothing else)."""
    message = f"[ALERT:{alert['type']}] {alert['message']}"
    print(message)
    logger.warning(message)


def check_drift_alerts(params: dict) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    try:
        numeric_report = check_drift(p_value_threshold=params["drift"]["ks_p_value_threshold"])
        drifted = numeric_report[numeric_report["drift"]] if not numeric_report.empty else numeric_report
        if len(drifted) > 0:
            alerts.append({
                "type": "DATA_DRIFT",
                "severity": "warning",
                "message": (f"Numeric data drift (KS test) detected in "
                            f"{len(drifted)} feature(s): {', '.join(drifted['feature'])}."),
                "detail": drifted.to_dict(orient="records"),
            })
    except FileNotFoundError as e:
        alerts.append({"type": "DATA_DRIFT", "severity": "info", "message": str(e), "detail": None})

    try:
        cat_report = check_categorical_drift(p_value_threshold=params["drift"]["categorical_p_value_threshold"])
        cat_drifted = cat_report[cat_report["drift"]] if not cat_report.empty else cat_report
        if len(cat_drifted) > 0:
            alerts.append({
                "type": "CATEGORICAL_DRIFT",
                "severity": "warning",
                "message": (f"Categorical data drift (chi-square test) detected in "
                            f"{len(cat_drifted)} feature(s): {', '.join(cat_drifted['feature'])}."),
                "detail": cat_drifted.to_dict(orient="records"),
            })
    except FileNotFoundError as e:
        alerts.append({"type": "CATEGORICAL_DRIFT", "severity": "info", "message": str(e), "detail": None})

    return alerts


def check_performance_alerts(params: dict) -> List[Dict[str, Any]]:
    result = check_performance_decay(params)
    if result["status"] == "DECAYED":
        return [{
            "type": "PERFORMANCE_DECAY",
            "severity": "critical",
            "message": f"Model performance has decayed: {result['reason']}",
            "detail": result,
        }]
    return []


def check_system_alerts(params: dict) -> List[Dict[str, Any]]:
    """API error-rate check. Reads the persisted prediction log (works even after the API
    process has restarted, unlike the in-process /metrics counters, which reset on
    restart)."""
    summary = summarize_predictions_log()
    total = summary.get("total_predictions", 0)
    if total == 0:
        return []
    error_count = summary.get("error_count", 0)
    error_rate = error_count / total
    threshold = params["alerting"]["api_error_rate_threshold"]
    if error_rate > threshold:
        return [{
            "type": "SYSTEM_ERROR_RATE",
            "severity": "critical",
            "message": (f"API error rate {error_rate:.1%} exceeds threshold {threshold:.1%} "
                        f"({error_count}/{total} predictions failed)."),
            "detail": summary,
        }]
    return []


def run_all_checks(params: dict | None = None) -> Dict[str, Any]:
    params = params or load_params()
    alerts: List[Dict[str, Any]] = []
    alerts += check_drift_alerts(params)
    alerts += check_performance_alerts(params)
    alerts += check_system_alerts(params)

    for alert in alerts:
        _dispatch_alert(alert)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_alerts": len(alerts),
        "alerts": alerts,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description="Run all threshold-based alert checks.")
    _ = parser.parse_args()
    report = run_all_checks()
    if report["n_alerts"] == 0:
        print("No alerts -- all checks within configured thresholds.")
    else:
        print(f"\n{report['n_alerts']} alert(s) raised -- see reports/alerts_report.json")


if __name__ == "__main__":
    main()
