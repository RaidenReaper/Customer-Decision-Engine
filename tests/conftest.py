from __future__ import annotations

import pytest

from src.generate_dummy_data import generate_dummy_data


@pytest.fixture(scope="session")
def dummy_data():
    return generate_dummy_data()


@pytest.fixture(scope="session")
def sample_input(dummy_data):
    return dummy_data.iloc[0].to_dict()
