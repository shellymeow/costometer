from pathlib import Path

import pandas as pd
import pytest

from costometer.utils.bias_utils import (  # noqa : F401
    calculate_overplanning,
    calculate_present_bias,
    calculate_present_bias_late,
)


# some setup where you load fake / simulate data as 'pytest fixture'
@pytest.fixture()
def bias_fixture():
    mouselab_data = pd.read_csv(
        Path(__file__).parents[0].joinpath("inputs/fake_data/mouselab_mdp.csv")
    )
    yield mouselab_data


def test_calculate_overplanning(bias_fixture):
    return NotImplementedError


def test_calculate_present_bias(bias_fixture):
    return NotImplementedError


def test_calculate_present_bias_late(bias_fixture):
    return NotImplementedError
