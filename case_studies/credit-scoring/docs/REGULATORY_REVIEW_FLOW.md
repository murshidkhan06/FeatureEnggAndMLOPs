# Regulatory review flow for a credit model

A credit model decides about people's money, so in most countries it cannot go live just because the data scientist is happy with its AUC. Banks follow a *model risk management* process (regulators call it SR 11-7 in the USA, SS1/23 in the UK, RBI / BCBS guidance in India and elsewhere - **rules differ by country, check yours**). This project reproduces the *shape* of that process with simple, checkable artefacts.

```
 1. DEVELOPMENT  ->  2. INDEPENDENT VALIDATION  ->  3. COMPLIANCE + FAIRNESS REVIEW
        |                                                          |
        v                                                          v
 6. MONITORING  <-  5. APPROVED FOR PRODUCTION  <-  4. MODEL RISK COMMITTEE
        |
        +--- drift / performance evidence ---> back to 1 (challenger) ---> review again
```

| Stage | Who | What they check | Artefact in this repo | Command |
|---|---|---|---|---|
| 1. Development | Data scientist + data engineer | Business problem, leakage-free features, simple baseline vs candidate, reproducible pipeline | feature registry, `reports/metrics.json`, MLflow runs, DVC pipeline | `python -m src.models.train` |
| 1b. Automated **review gate** | CI / model owner | Hard rules (below) - the gate says GO or NO-GO *before* humans spend time | `src/governance/review_gate.py` | `python -m src.governance.review_gate` |
| 2. Independent validation | Validation team (not the builders) | Re-compute metrics, challenge assumptions, stability (KS train vs test), calibration, challenger comparison | `reports/model_review_pack.md` sections 5-6 and 8 | `python -m src.governance.model_card` |
| 3. Compliance + fairness | Compliance officer | Sensitive / restricted features, proxies, adverse impact ratio, adverse-action reason codes | fairness audit (review pack section 7), age experiment, registry `status`/`reason` | `python -m src.governance.fairness_audit` |
| 4. Model risk committee | Senior risk, business, audit | Limitations, monitoring plan, sign-off checklist | review pack sections 9-11 (limitations, monitoring plan, sign-off) | (meeting; the checklist stays PENDING until humans sign) |
| 5. Approved for production | Model owner | Deploy the *approved* artefact; record version and approvers | `models/registry.json`, approval record from `retrain.py` | `python -m src.models.retrain --approve-as "NAME (ROLE)"` |
| 6. Monitoring | ML engineer + risk | Input drift, labelled performance (12 months later), calibration, fairness drift, data quality | drift check, Prometheus alerts, decision ladder | `python -m src.monitoring.run_drift_check` |

## The automated gate (stage 1b) fails when ANY of these is true
1. a feature used by the model is not **Approved** in the feature registry (leakage, sensitive, restricted, unreviewed proxy);
2. a sensitive / restricted / proxy attribute (gender, marital status, age, fine location) is inside the model;
3. the **adverse impact ratio** (female approval rate / male approval rate) is below **0.8** (the "four-fifths rule", a screening heuristic, not a legal verdict);
4. the model card / review pack is missing or incomplete;
5. test **ROC-AUC** is below the minimum (0.72);
6. the model cannot produce **reason codes** (every declined or referred applicant is entitled to know the main reasons: this is why a linear scorecard is preferred over a black box).

`python -m src.governance.review_gate --demo-bad-model` shows the gate **rejecting a leaky, sensitive-feature, black-box model that has a "great" AUC of 0.95**.

## Policy decisions that are NOT technical
* **Age**: many jurisdictions restrict or forbid its use in credit scoring. We measured "with age vs without age" (AUC change is within noise, young applicants' approval drops with age) and **excluded it by default**. Compliance owns the decision; changing it needs a new registry version and a new review.
* **Location**: fine-grained location behaves as a proxy for gender in this data; it is excluded (the coarse `city_tier` carries no extra signal here: the selection step rejected it as weak).
* **Automatic action**: a person - not a script - signs the promotion. `retrain.py` prints an approval record and waits for `--approve-as`.

## What humans sign (checklist at the end of the review pack)
Model owner - Independent validation - Compliance - Model risk committee. Status values: PENDING / APPROVED / REJECTED. The generator never fills them in.
