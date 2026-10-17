"""Evaluate the CHAMPION on the held-out TEST period and translate the errors into business language.

Run:  python -m src.models.evaluate            (writes reports/metrics.json for DVC)

Reading the numbers (store-manager language)
  WAPE   0.13 -> "for every 100 units we sold, our forecast was off by about 13 units"
  bias  -2%   -> "over the quarter we forecast 2% FEWER units than we sold: a slight tendency to under-stock"
  over-forecast units  -> stock that was ordered but not sold   (waste, holding cost)
  under-forecast units -> demand we could not serve             (lost sales, unhappy customers)
  weighted cost = over x 1 + under x 3  (under-forecasting hurts ~3x more per unit; set in configs/params.yaml)
"""
from __future__ import annotations

import json

import pandas as pd

from src.common.artifact import load_artifact
from src.common.facts import save_fact
from src.common.params import load_params
from src.common.paths import CHAMPION_PATH, DATA_RAW, REPORTS
from src.features.build_features import training_frame
from src.models import metrics as M
from src.models.baselines import BASELINES, baseline_forecast
from src.models.train import score, time_split


def main() -> None:
    params = load_params()
    art = load_artifact(CHAMPION_PATH)
    feats = art["input_columns"]
    raw = pd.read_csv(DATA_RAW / "sales_daily.csv", parse_dates=["date"])
    df = training_frame(raw)
    _, valid, test = time_split(df, params["split"])
    costs = params["costs"]
    pd.set_option("display.width", 160)

    print(f"CHAMPION {art['version']} ({art['extra']['model_name']}), trained through {art['extra']['trained_through']}, "
          f"features: {len(feats)}")
    print(f"TEST period {test.date.min().date()} .. {test.date.max().date()}: {len(test):,} rows, "
          f"{test.sales.sum():,.0f} units sold\n")

    rows, preds = {}, {}
    for name in BASELINES:
        preds[name] = baseline_forecast(name, test).values
    preds["champion"] = score(art["pipeline"], test, feats)
    for name, f in preds.items():
        rows[name] = {**M.report(test["sales"], f, test["rolling_mean_30"]), **M.cost_report(test["sales"], f, costs["over_forecast"], costs["under_forecast"])}
    tbl = pd.DataFrame(rows).T
    print("1) ACCURACY AND BUSINESS COST ON THE TEST PERIOD (lower is better except roc_auc)")
    print(tbl.to_string())
    ch, sn = rows["champion"], rows["seasonal_naive"]
    print(f"\n   Champion vs seasonal naive: WAPE {sn['wape']:.3f} -> {ch['wape']:.3f}  "
          f"({1 - ch['wape'] / sn['wape']:.0%} fewer errors); weighted stock cost {sn['weighted_cost']:,} -> {ch['weighted_cost']:,}")
    print(f"   Meaning: of {test.sales.sum():,.0f} units sold, the forecast over-stated {ch['over_forecast_units']:,} unit-days and "
          f"missed {ch['under_forecast_units']:,} unit-days of demand;")
    print(f"   bias {ch['bias_pct']:+.1%} means the forecast was {'slightly LOW (risk of stock-outs)' if ch['bias_pct'] < 0 else 'slightly HIGH (risk of waste)'} overall.")
    print(f"   roc_auc = {ch['roc_auc']:.3f}: how well the forecast RANKS 'spike' days (sales > 1.15 x recent level) above normal days"
          " (0.5 = coin flip).")

    print("\n2) WHERE ARE THE ERRORS?  (champion WAPE by segment)")
    t = test.assign(f=preds["champion"])
    seg = {}
    for label, mask in (("promotion day", t.promotion_flag == 1), ("no promotion", t.promotion_flag == 0),
                        ("holiday", t.is_holiday == 1), ("weekend", t.is_weekend == 1), ("weekday", t.is_weekend == 0)):
        seg[label] = dict(rows=int(mask.sum()), wape=round(M.wape(t.sales[mask], t.f[mask]), 3), bias_pct=round(M.bias_pct(t.sales[mask], t.f[mask]), 3))
    print(pd.DataFrame(seg).T.to_string())
    by_p = t.groupby("product_id").apply(lambda g: pd.Series(dict(wape=M.wape(g.sales, g.f), bias_pct=M.bias_pct(g.sales, g.f))), include_groups=False)
    print("\n   by product:\n" + by_p.round(3).T.to_string())

    print("\n3) PREDICTION INTERVAL (OPTIONAL): calibrated on validation residuals, checked on test")
    q = art["extra"]["interval"]
    print(f"   [{q['low']:+.2f}, {q['high']:+.2f}] x forecast  -> nominal {q['level']:.0%} coverage; observed on test = {art['metrics']['coverage']:.1%}")

    print("\n4) WHY NOT MAPE?  Zero-sales (stock-out) days  (all 2 years of rows, seasonal-naive forecast)")
    z = int((df.sales == 0).sum())
    mape_all = M.mape_naive(df.sales, df.lag_7)
    print(f"   {z} zero-sales rows -> MAPE = {mape_all} (division by zero: one stock-out ruins the average);")
    print(f"   WAPE = {M.wape(df.sales, df.lag_7):.4f} is always defined because it divides by TOTAL units, not by each day.")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(dict(champion_version=art["version"], model=art["extra"]["model_name"],
                                                          test=rows["champion"], baselines={k: v for k, v in rows.items() if k != "champion"}), indent=2))
    save_fact("evaluation", dict(table=tbl.round(4).reset_index(names="model").to_dict("records"), segments=seg,
                                 by_product=by_p.round(4).to_dict("index"), interval=q, zero_rows_test=z))


if __name__ == "__main__":
    main()
