"""FEATURE LIFECYCLE GOVERNANCE: who may use which feature, and why.

WHY
  In a bank, "I computed a nice column in my notebook" is not enough.  Model-risk regulations expect every model input to
  have an owner, a definition, a known availability at prediction time, and a documented approval.  The feature registry
  (configs/feature_registry.yaml) is that paper trail, and THIS module turns the paper trail into code that can block a build.

LIFECYCLE
        Development -> Validation -> Approval -> Production           (Rejected is a dead end;  Retired = switched off)

  Gate rules (validation_rules in the registry):
     1. leakage_test           the future-perturbation test passed (features do not change when the future is corrupted)
     2. null_rate              share of missing values <= max_null_rate (0.20)
     3. availability rule      the feature declares its availability lag (when is the data usable?)
     4. owner                  somebody is accountable for it
  A feature may be APPROVED or PUT INTO PRODUCTION only if all four checks passed.

  Training refuses to use anything that is not (status == Approved and stage == Production):  check_features_approved().

TINY EXAMPLE
  check_features_approved(["average_balance", "average_balance_leaky"])
  -> FeatureNotApprovedError: average_balance_leaky v1 is Rejected (Leakage: includes the post-default balance collapse ...)

CLI
  python -m src.governance.feature_lifecycle table       print the registry as a table
  python -m src.governance.feature_lifecycle history     print the lifecycle history of every feature
  python -m src.governance.feature_lifecycle validate    run the REAL validation checks on today's data and compare
  python -m src.governance.feature_lifecycle gate-demo   show a feature being blocked from Production
"""
from __future__ import annotations

import argparse
import copy

import yaml

from src.common.paths import FEATURE_REGISTRY_PATH

STAGES = ["Development", "Validation", "Approval", "Production"]
ALLOWED = {"Development": ["Validation"], "Validation": ["Approval", "Rejected"], "Approval": ["Production", "Rejected"],
           "Production": ["Retired"], "Rejected": [], "Retired": []}


class FeatureNotApprovedError(Exception):
    """Raised when training/serving asks for a feature that is not Approved + Production."""


class LifecycleError(Exception):
    """Raised for an illegal lifecycle transition (e.g. skipping Validation, or failing a gate check)."""


# ----------------------------------------------------------------------------------------------------------
def load_registry(path=None) -> dict:
    """Read configs/feature_registry.yaml into a dict."""
    return yaml.safe_load(open(path or FEATURE_REGISTRY_PATH, encoding="utf-8"))


def latest_entries(registry: dict) -> dict:
    """{feature name: entry with the highest version}.  Example: average_balance -> version 2."""
    out: dict = {}
    for f in registry["features"]:
        if f["name"] not in out or f["version"] > out[f["name"]]["version"]:
            out[f["name"]] = f
    return out


def check_features_approved(features: list[str], registry: dict | None = None) -> list[str]:
    """Raise FeatureNotApprovedError unless EVERY feature is Approved + in Production (latest version).  Returns the list."""
    registry = registry or load_registry()
    entries = latest_entries(registry)
    problems = []
    for name in features:
        e = entries.get(name)
        if e is None:
            problems.append(f"{name}: not in the feature registry (unregistered features are not allowed)")
        elif e["status"] != "Approved" or e["stage"] != "Production":
            why = e.get("rejection_reason", f"stage is {e['stage']}")
            problems.append(f"{name} v{e['version']} is {e['status']} / {e['stage']} ({why})")
    if problems:
        raise FeatureNotApprovedError("Training refused. " + " | ".join(problems))
    return list(features)


def validation_failures(feature: dict, rules: dict, evidence: dict | None = None) -> list[str]:
    """List the gate checks that FAIL for this feature (empty list = all pass).

    `evidence` = freshly measured values {'leakage_test': 'pass'|'FAIL', 'null_rate': float}; if omitted we use the
    checks recorded in the feature's Validation history entry."""
    recorded = next((h.get("checks", {}) for h in reversed(feature.get("history", [])) if h["stage"] == "Validation"), {})
    ev = {**recorded, **(evidence or {})}
    fails = []
    if rules.get("require_leakage_test_pass", True) and ev.get("leakage_test") != "pass":
        fails.append(f"leakage_test = {ev.get('leakage_test', 'not run')}")
    if ev.get("null_rate") is None or ev["null_rate"] > rules.get("max_null_rate", 0.20):
        fails.append(f"null_rate = {ev.get('null_rate')} > {rules.get('max_null_rate', 0.20)}")
    av = feature.get("availability", {})
    if rules.get("require_availability_rule", True) and (av.get("lag_days") is None or not av.get("rule", "").startswith("available_at")):
        fails.append("availability rule not declared")
    if rules.get("require_owner", True) and not feature.get("owner"):
        fails.append("no owner assigned")
    return fails


def advance(feature: dict, to_stage: str, by: str, date: str, rules: dict, evidence: dict | None = None) -> dict:
    """Move a feature one step through the lifecycle and return the UPDATED COPY (the input is not modified).

    Illegal moves (skipping a stage) and failed gate checks raise LifecycleError."""
    cur = feature["stage"]
    if to_stage not in ALLOWED.get(cur, []):
        raise LifecycleError(f"{feature['name']}: cannot go {cur} -> {to_stage} (allowed: {ALLOWED.get(cur)})")
    if to_stage in ("Approval", "Production"):
        fails = validation_failures(feature, rules, evidence)
        if fails:
            raise LifecycleError(f"{feature['name']}: gate to {to_stage} blocked - " + "; ".join(fails))
    new = copy.deepcopy(feature)
    new["stage"] = to_stage
    new["status"] = "Rejected" if to_stage == "Rejected" else "Approved" if to_stage in ("Approval", "Production") else new["status"]
    entry = {"stage": to_stage, "date": date, "by": by}
    if evidence:
        entry["checks"] = evidence
    new["history"].append(entry)
    return new


# ----------------------------------------------------------------------------------------------------------
def registry_table(registry: dict | None = None):
    """The registry as a pandas DataFrame (one row per feature version)."""
    import pandas as pd
    registry = registry or load_registry()
    rows = []
    for f in registry["features"]:
        last = f["history"][-1]
        rows.append(dict(feature=f["name"], ver=f["version"], status=f["status"], stage=f["stage"], owner=f["owner"],
                         lag_days=f["availability"]["lag_days"], window=f["window"], source=f["source"],
                         last_event=f"{last['stage']} {last['date']} by {last['by']}"))
    return pd.DataFrame(rows)


def _print_table(registry: dict) -> None:
    import pandas as pd
    pd.set_option("display.width", 250, "display.max_colwidth", 48)
    print(registry_table(registry).to_string(index=False))


def _print_history(registry: dict) -> None:
    for f in registry["features"]:
        print(f"\n{f['name']} v{f['version']}  [{f['status']} / {f['stage']}]  owner={f['owner']}")
        for h in f["history"]:
            chk = ""
            if "checks" in h:
                c = h["checks"]
                chk = f"  checks: leakage={c['leakage_test']} null={c['null_rate']} availability={c['availability_rule_declared']} owner={c['owner_assigned']}"
            print(f"   {h['date']}  {h['stage']:<11} {h['by']:<22} {h.get('note', '')}{chk}")


def _validate_live(registry: dict) -> None:
    """Re-run the REAL checks on the current data and show whether each feature would pass the Production gate today."""
    import pandas as pd
    from src.common.config import load_params
    from src.common.paths import SNAPSHOTS_PATH
    from src.features.build_features import add_derived_features, build_snapshot_features, load_raw
    from src.features.definitions import BASE_FEATURES, MODEL_FEATURES
    from src.features.leakage_checks import future_perturbation_test
    from src.features.leakage_demo import LEAKY_BUILDERS

    raw = load_raw()
    snap = add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
    train = snap[snap["split"] == "train"]
    nulls = train[MODEL_FEATURES].isna().mean()
    as_ofs = [pd.Timestamp("2022-06-30"), pd.Timestamp("2022-12-31")]

    def safe_builder(c, t, l, a):
        return add_derived_features(build_snapshot_features(c, t, l, a))[["customer_id"] + MODEL_FEATURES]

    changed: set = set()
    for a in as_ofs:
        changed |= set(future_perturbation_test(safe_builder, raw, a)["changed_columns"])
    for name, fn in LEAKY_BUILDERS.items():
        if not future_perturbation_test(fn, raw, as_ofs[0])["passed"]:
            changed.add(name)
    rules = registry["validation_rules"]
    rows = []
    for f in latest_entries(registry).values():
        n = f["name"]
        ev = {"leakage_test": "FAIL" if n in changed else "pass", "null_rate": round(float(nulls.get(n, 0.0)), 4)}
        fails = validation_failures(f, rules, ev)
        if not fails and f["status"] == "Rejected":
            fails = ["Rejected by the Model Risk Committee (a policy decision, not a technical check)"]
        rows.append(dict(feature=n, ver=f["version"], leakage_test=ev["leakage_test"], null_rate=ev["null_rate"],
                         availability_declared=f["availability"]["lag_days"] is not None, owner=bool(f["owner"]),
                         may_enter_production="YES" if not fails else "NO", registry_status=f["status"],
                         why_not="; ".join(fails)))
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(pd.DataFrame(rows).to_string(index=False))


def _gate_demo(registry: dict) -> None:
    """Show the gate blocking a feature, then letting a good one through (in memory only)."""
    rules = registry["validation_rules"]
    entries = latest_entries(registry)
    leaky = copy.deepcopy(entries["average_balance_leaky"])
    leaky["stage"] = "Validation"          # pretend it is waiting for approval
    leaky["status"] = "Approved"
    print("1) Try to approve average_balance_leaky (its Validation record says leakage_test FAIL):")
    try:
        advance(leaky, "Approval", "Model Risk Committee", "2022-10-11", rules)
    except LifecycleError as e:
        print("   BLOCKED ->", e)
    good = copy.deepcopy(entries["average_balance"])
    good["stage"] = "Development"
    print("2) Try to jump average_balance straight from Development to Production:")
    try:
        advance(good, "Production", "MLOps team", "2022-11-21", rules)
    except LifecycleError as e:
        print("   BLOCKED ->", e)
    print("3) Legal path for a clean feature:")
    ev = {"leakage_test": "pass", "null_rate": 0.0, "availability_rule_declared": True, "owner_assigned": True}
    for step in ("Validation", "Approval", "Production"):
        good = advance(good, step, "demo", "2022-11-21", rules, ev if step == "Validation" else None)
        print(f"   -> {good['stage']}")
    print("4) Training-time check:")
    for names in (["average_balance", "salary_credit_frequency"], ["average_balance_leaky"], ["segment"]):
        try:
            check_features_approved(names, registry)
            print(f"   {names}: OK")
        except FeatureNotApprovedError as e:
            print(f"   {names}: REFUSED -> {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("CLI")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["table", "history", "validate", "gate-demo"])
    a = ap.parse_args()
    reg = load_registry()
    {"table": _print_table, "history": _print_history, "validate": _validate_live, "gate-demo": _gate_demo}[a.command](reg)
