import itertools
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from mouselab.cost_functions import linear_depth
from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from mouselab.policies import RandomPolicy, SoftmaxPolicy

from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference.grid import GridInference
from costometer.utils import load_q_file, save_q_values_for_cost

exp_settings = [
    {"setting": "small_increasing", "reward_dictionary": high_increasing_reward},
    {"setting": "small_decreasing", "reward_dictionary": high_decreasing_reward},
]

cost_value_combinations = [
    {"depth_cost_weight": 0, "static_cost_weight": 0},
    {"depth_cost_weight": 10, "static_cost_weight": 10},
]

mle_test_data = [
    {
        "env": exp_setting,
        "num_episodes": 30,
        "cost_input": {"depth_cost_weight": 0, "static_cost_weight": 1},
        "policy_function": RandomPolicy,
        "policy_kwargs": {},
    }
    for exp_setting in exp_settings
]

for exp_setting in exp_settings:
    for cost_value_combination in cost_value_combinations:
        mle_test_data.append(
            {
                "env": exp_setting,
                "num_episodes": 30,
                "cost_input": cost_value_combination,
                "policy_function": SoftmaxPolicy,
                "policy_kwargs": {"preference": {}, "temp": 1, "noise": 0},
            },
        )

inference_cost_parameters = {
    "depth_cost_weight": Categorical([0, 1, 10]),
    "static_cost_weight": Categorical([0, 1, 10]),
}


# this time we want two: one for small_increasing and one for small_decreasing
# in each, all three traces are evaluated
@pytest.fixture(params=mle_test_data)
def mle_test_cases(request, inference_cost_parameters=inference_cost_parameters):
    # build case, first registering environment
    register(
        name=request.param["env"]["setting"],
        branching=[2, 2],
        reward_inputs="depth",
        reward_dictionary=request.param["env"]["reward_dictionary"],
    )

    # load q files for inference
    input_path = Path(__file__).parents[0].joinpath("./inputs/q_files")
    input_path.mkdir(parents=True, exist_ok=True)

    keys, vals = zip(*inference_cost_parameters.items())
    for curr_cost_parameters in [
        dict(zip(keys, curr_cost))
        for curr_cost in list(itertools.product(*[val.vals for val in vals]))
    ]:
        try:
            load_q_file(
                request.param["env"]["setting"],
                cost_function=linear_depth,
                cost_params=curr_cost_parameters,
                path=input_path,
            )
        except IndexError:
            save_q_values_for_cost(
                request.param["env"]["setting"],
                cost_function=linear_depth,
                cost_params=curr_cost_parameters,
                path=input_path,
            )

    q_path = Path(__file__).parents[0].joinpath("./inputs/q_files")
    q_path.mkdir(parents=True, exist_ok=True)
    # load q file
    if request.param["policy_function"] != RandomPolicy:
        # create Q values if needed
        try:
            q_dictionary = load_q_file(
                request.param["env"]["setting"],
                cost_function=linear_depth,
                cost_params=request.param["cost_input"],
                path=q_path,
            )
        except IndexError:
            q_dictionary = save_q_values_for_cost(
                request.param["env"]["setting"],
                cost_function=linear_depth,
                cost_params=request.param["cost_input"],
                path=q_path,
            )

        request.param["policy_kwargs"]["preference"] = q_dictionary

    agent = SymmetricMouselabParticipant(
        request.param["env"]["setting"],
        num_trials=request.param["num_episodes"],
        cost_function=linear_depth,
        cost_kwargs=request.param["cost_input"],
        policy_function=request.param["policy_function"],
        policy_kwargs=request.param["policy_kwargs"],
    )

    trace = agent.simulate_trajectory()
    trace["pid"] = [0] * len(trace["states"])

    inference_policy_kwargs = deepcopy(request.param["policy_kwargs"])
    if "preference" in inference_policy_kwargs:
        del inference_policy_kwargs["preference"]
    inference_policy_kwargs["q_path"] = q_path

    softmax_inference_agent_kwargs = {
        "participant_class": SymmetricMouselabParticipant,
        "participant_kwargs": {
            "experiment_setting": request.param["env"]["setting"],
            "policy_function": SoftmaxPolicy,
        },
        # {"num_trials" : request.param["num_episodes"]},
        "cost_function": linear_depth,
        "cost_parameters": inference_cost_parameters,
        "held_constant_policy_kwargs": inference_policy_kwargs,
    }

    random_inference_agent_kwargs = {
        "participant_class": SymmetricMouselabParticipant,
        "participant_kwargs": {
            "experiment_setting": request.param["env"]["setting"],
            "policy_function": RandomPolicy,
        },
        # {"num_trials" : request.param["num_episodes"]},
        "cost_function": linear_depth,
        "cost_parameters": {
            "depth_cost_weight": Categorical([None]),
            "static_cost_weight": Categorical([None]),
        },
        "held_constant_policy_kwargs": {},
    }

    if request.param["policy_function"] != RandomPolicy:
        correct_inference = request.param["cost_input"]
    else:
        correct_inference = {key: None for key in request.param["cost_input"].keys()}

    yield [
        trace
    ], softmax_inference_agent_kwargs, random_inference_agent_kwargs, correct_inference

    # cleanup if needed
    pass


def test_instantiate(mle_test_cases):
    traces, softmax_inference_agent_kwargs, _, _ = mle_test_cases
    mle_algorithm = GridInference(traces, **softmax_inference_agent_kwargs)
    assert isinstance(mle_algorithm, GridInference)


def test_mle_run(mle_test_cases):
    (
        traces,
        softmax_inference_agent_kwargs,
        random_inference_agent_kwargs,
        correct_inference,
    ) = mle_test_cases
    softmax_mle_algorithm = GridInference(traces, **softmax_inference_agent_kwargs)
    random_mle_algorithm = GridInference(traces, **random_inference_agent_kwargs)

    # run algorithm
    softmax_mle_algorithm.run()
    random_mle_algorithm.run()

    results = pd.concat(
        [
            random_mle_algorithm.get_optimization_results(),
            softmax_mle_algorithm.get_optimization_results(),
        ],
        ignore_index=True,
    )

    for key, val in correct_inference.items():
        assert results.loc[results["map"].idxmax(), key] == val
