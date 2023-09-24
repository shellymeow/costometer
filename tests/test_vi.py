import pytest
from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from mouselab.exact_utils import timed_solve_env

from costometer.envs.modified_mouselab import ModifiedMouseLabEnv
from costometer.planning_algorithms.vi import flatten_q, value_iteration

vi_test_data = [
    {
        "name": "medium_magellanic_cloud",
        "branching": [1, 2, 1],
        "reward_inputs": "depth",
        "reward_dictionary": {
            1: Categorical([-500]),
            2: Categorical([-60, 60]),
            3: Categorical([-90, 90]),
        },
    },
    {
        "name": "small_increasing",
        "branching": [2, 2],
        "reward_inputs": "depth",
        "reward_dictionary": high_increasing_reward,
    },
    {
        "name": "small_decreasing",
        "branching": [2, 2],
        "reward_inputs": "depth",
        "reward_dictionary": high_decreasing_reward,
    },
]


@pytest.fixture(params=vi_test_data)
def vi_test_cases(request):
    register(**request.param)
    yield request.param["name"]


def test_value_iteration(vi_test_cases):
    setting = vi_test_cases
    env = ModifiedMouseLabEnv.new_symmetric_registered(setting)

    Q, _, _, _ = value_iteration(env)
    _, _, _, info = timed_solve_env(env, save_q=True)

    for state in env.P.keys():
        for action in env.actions(state):
            assert info["q_dictionary"][(state, action)] == Q[state][action]


def test_flatten_q(vi_test_cases):
    setting = vi_test_cases
    env = ModifiedMouseLabEnv.new_symmetric_registered(setting)

    Q, _, _, _ = value_iteration(env)
    _, _, _, info = timed_solve_env(env, save_q=True)

    flattened_q = flatten_q(Q)

    for key, val in flattened_q.items():
        assert info["q_dictionary"][key] == val
