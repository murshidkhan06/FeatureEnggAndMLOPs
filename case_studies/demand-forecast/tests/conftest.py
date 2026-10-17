"""Shared fixtures.  Data is generated in memory (seeded, ~1 s) so the data/feature tests need NO files from earlier steps."""
import pytest

from src.data import make_dataset as MD
from src.features.build_features import training_frame


@pytest.fixture(scope="session")
def datasets():
    return MD.generate_all()


@pytest.fixture(scope="session")
def raw(datasets):
    return datasets["train"]


@pytest.fixture(scope="session")
def train_frame(raw):
    return training_frame(raw)
