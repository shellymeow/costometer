from pathlib import Path
from shutil import rmtree

import pytest
from mouselab.cost_functions import linear_depth
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward

from costometer.utils import save_q_values_for_cost

from .fixtures import load_structure_dicts

save_q_test_data = [
    {
        "env": {
            "name": "small_increasing",
            "branching": [2, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        },
        "cost_kwargs": {
            "cost_function": linear_depth,
            "cost_params": {"depth_cost_weight": 10, "static_cost_weight": 10},
        },
    },
    {
        "env": {
            "name": "small_decreasing",
            "branching": [2, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_decreasing_reward,
        },
        "cost_kwargs": {
            "cost_function": linear_depth,
            "cost_params": {"depth_cost_weight": 0, "static_cost_weight": 1},
        },
    },
]


@pytest.fixture(params=save_q_test_data)
def save_q_test_cases(request):
    # setup
    output_path = Path(__file__).parents[0].joinpath("outputs")
    rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    register(**request.param["env"])

    yield request.param["env"]["name"], output_path, request.param["cost_kwargs"]

    rmtree(output_path, ignore_errors=True)


def test_save_q_values_for_cost(save_q_test_cases):
    structure_dicts = load_structure_dicts("small_structure")

    experiment_setting, path, cost_kwargs = save_q_test_cases
    beginning_file_num = len(
        list(
            (path / experiment_setting / cost_kwargs["cost_function"].__name__).glob(
                "*"
            )
        )
    )
    save_q_values_for_cost(
        experiment_setting, path=path, structure=structure_dicts, **cost_kwargs
    )
    ending_file_num = len(
        list(
            (path / experiment_setting / cost_kwargs["cost_function"].__name__).glob(
                "*"
            )
        )
    )
    # just check file is being saved
    assert ending_file_num - beginning_file_num == 1
