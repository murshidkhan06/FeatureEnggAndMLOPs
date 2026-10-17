"""Shared fixtures.  Tests that need generated data / a trained champion skip with a clear message if you have not run the pipeline."""
import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd
import pytest

from src.common.paths import CHAMPION_PATH, DATA_PROCESSED, DATA_PRODUCTION, DATA_RAW

needs_data = pytest.mark.skipif(not (DATA_PROCESSED / "train.csv").exists(), reason="run: python -m src.data.make_dataset && python -m src.features.build_features")
needs_model = pytest.mark.skipif(not CHAMPION_PATH.exists(), reason="run: python -m src.models.train --bootstrap-champion")


@pytest.fixture(scope="session")
def small_raw():
    """A small deterministic raw dataset generated in memory (does not touch data/)."""
    from src.data.make_dataset import generate
    return generate(n=4000, seed=123)


@pytest.fixture(scope="session")
def splits():
    return tuple(pd.read_csv(DATA_PROCESSED / f"{n}.csv") for n in ("train", "valid", "test"))


@pytest.fixture(scope="session")
def champion():
    from src.common.artifact import load_artifact
    return load_artifact(CHAMPION_PATH)


@pytest.fixture(scope="session")
def batches():
    return pd.read_csv(DATA_PRODUCTION / "batch_normal.csv"), pd.read_csv(DATA_PRODUCTION / "batch_drifted.csv")
