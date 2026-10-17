"""Shared fixtures.  The raw tables are re-generated in memory (about 4 seconds, seeded) so that unit tests do not depend on files on disk."""
import pandas as pd
import pytest

from src.common.paths import CHAMPION_PATH, DATA_PRODUCTION, REFERENCE_PATH, SNAPSHOTS_PATH
from src.data.make_dataset import generate


@pytest.fixture(scope="session")
def raw():
    return generate()


needs_snapshots = pytest.mark.skipif(not SNAPSHOTS_PATH.exists(), reason="run src.features.build_features first")
needs_champion = pytest.mark.skipif(not CHAMPION_PATH.exists(), reason="train a model first")
needs_production = pytest.mark.skipif(not (DATA_PRODUCTION / "batch_normal.csv").exists() or not REFERENCE_PATH.exists(),
                                      reason="run src.features.build_features first")


@pytest.fixture(scope="session")
def snapshots():
    from src.features.build_features import add_derived_features
    return add_derived_features(pd.read_csv(SNAPSHOTS_PATH, parse_dates=["as_of_date"]))
