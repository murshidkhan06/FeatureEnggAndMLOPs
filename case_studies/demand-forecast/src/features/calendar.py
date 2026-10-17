"""Holiday calendar = information that is KNOWN IN ADVANCE.

WHY this file exists
--------------------
A forecast for tomorrow may use anything we already know about tomorrow:
the holiday calendar, the planned price, the planned promotion, the weather *forecast*.
It may NOT use tomorrow's sales or tomorrow's *actual* weather.  The holiday calendar is the
cleanest example of a "known in advance" feature, because it is published a year ahead.

The dates below are the (approximate) Indian retail holidays used by the synthetic data generator
and by the batch forecaster.  Diwali gets its own list because the data generator adds a
"festival lead-up" ramp in the 7 days before it (shoppers stock up) - this is the
EXPECTED seasonal shift used in the drift lessons.

Tiny worked example
-------------------
>>> is_holiday("2024-11-01")      # Diwali 2024
1
>>> is_holiday("2024-11-02")
0
"""
from __future__ import annotations

import pandas as pd

# date -> (name, strength).  strength = how much the day lifts demand for an "average" product.
HOLIDAYS: dict[str, tuple[str, float]] = {
    "2023-01-01": ("New Year", 1.15), "2023-01-26": ("Republic Day", 1.15), "2023-03-08": ("Holi", 1.30),
    "2023-04-22": ("Eid", 1.25), "2023-08-15": ("Independence Day", 1.15), "2023-08-30": ("Raksha Bandhan", 1.25),
    "2023-10-24": ("Dussehra", 1.25), "2023-11-12": ("Diwali", 1.60), "2023-12-25": ("Christmas", 1.30),
    "2024-01-01": ("New Year", 1.15), "2024-01-26": ("Republic Day", 1.15), "2024-03-25": ("Holi", 1.30),
    "2024-04-11": ("Eid", 1.25), "2024-08-15": ("Independence Day", 1.15), "2024-08-19": ("Raksha Bandhan", 1.25),
    "2024-10-12": ("Dussehra", 1.25), "2024-11-01": ("Diwali", 1.60), "2024-12-25": ("Christmas", 1.30),
    "2025-01-01": ("New Year", 1.15), "2025-01-26": ("Republic Day", 1.15), "2025-03-14": ("Holi", 1.30),
    "2025-03-31": ("Eid", 1.25), "2025-08-09": ("Raksha Bandhan", 1.25), "2025-08-15": ("Independence Day", 1.15),
    "2025-10-02": ("Dussehra", 1.25), "2025-10-20": ("Diwali", 1.60), "2025-12-25": ("Christmas", 1.30),
}
DIWALI_DATES = [pd.Timestamp(d) for d, (n, _) in HOLIDAYS.items() if n == "Diwali"]


def is_holiday(date) -> int:
    """1 if the date is in the holiday calendar, else 0."""
    return int(pd.Timestamp(date).strftime("%Y-%m-%d") in HOLIDAYS)


def holiday_strength(date) -> float:
    """Demand lift of the holiday for an average product (1.0 = no lift)."""
    return HOLIDAYS.get(pd.Timestamp(date).strftime("%Y-%m-%d"), ("", 1.0))[1]
