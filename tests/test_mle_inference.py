import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from mouselab.cost_functions import linear_depth
from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import RandomPolicy, SoftmaxPolicy

from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference.grid import GridInference
from costometer.utils import get_state_action_values

mle_test_data = [
    {
        "env": {
            "setting": "small_increasing",
            "reward_dictionary": high_increasing_reward,
        },
        "num_episodes": 30,
        "cost_input": {"depth_cost_weight": 0, "static_cost_weight": 1},
        "policy_function": RandomPolicy,
        "policy_kwargs": {},
    },
    {
        "env": {
            "setting": "small_decreasing",
            "reward_dictionary": high_decreasing_reward,
        },
        "num_episodes": 30,
        "cost_input": {"depth_cost_weight": 0, "static_cost_weight": 1},
        "policy_function": RandomPolicy,
        "policy_kwargs": {},
    },
]

# Trace cost value combination
cost_value_combinations = [
    {"depth_cost_weight": 0, "static_cost_weight": 0},
    {"depth_cost_weight": 10, "static_cost_weight": 10},
]

for cost_value_combination in cost_value_combinations:
    mle_test_data.extend(
        [
            {
                "env": {
                    "setting": "small_increasing",
                    "reward_dictionary": high_increasing_reward,
                },
                "num_episodes": 30,
                "cost_input": cost_value_combination,
                "policy_function": SoftmaxPolicy,
                "policy_kwargs": {"temp": 1, "noise": 0},
            },
            {
                "env": {
                    "setting": "small_decreasing",
                    "reward_dictionary": high_decreasing_reward,
                },
                "num_episodes": 30,
                "cost_input": cost_value_combination,
                "policy_function": SoftmaxPolicy,
                "policy_kwargs": {"temp": 1, "noise": 0},
            },
        ]
    )


@pytest.fixture(params=mle_test_data)
def mle_test_cases(request, inference_cost_parameters=None):
    if inference_cost_parameters is None:
        inference_cost_parameters = {
            "depth_cost_weight": Categorical([0, 1, 10]),
            "static_cost_weight": Categorical([0, 1, 10]),
        }

    # build case, first registering environment
    register(
        name=request.param["env"]["setting"],
        branching=[2, 2],
        reward_inputs="depth",
        reward_dictionary=request.param["env"]["reward_dictionary"],
    )

    # load structure file
    structure_file = (
        Path(__file__).parents[0] / "inputs" / "structure" / "small_structure.json"
    )
    with open(
        structure_file,
        "rb",
    ) as f:
        structure_data = json.load(f)

    structure_dicts = get_structure_properties(structure_data)
    env_params = {"include_last_action": True, "last_action_info": "distance"}
    additional_mouselab_kwargs = {
        "mdp_graph_properties": structure_dicts,
        **env_params,
    }

    # load q function
    q_function = None
    if request.param["policy_function"] != RandomPolicy:
        q_function = get_state_action_values(  # noqa : E731
            experiment_setting=request.param["env"]["setting"],
            cost_function=linear_depth,
            cost_parameters=request.param["cost_input"],
            bmps_file="Myopic_VOC",
            structure=structure_dicts,
            env_params=env_params,
            bmps_path=Path(__file__).parents[0].joinpath("inputs/parameters/bmps/"),
        )
        request.param["policy_kwargs"]["preference"] = q_function

    agent = SymmetricMouselabParticipant(
        request.param["env"]["setting"],
        num_trials=request.param["num_episodes"],
        cost_function=linear_depth,
        cost_kwargs=request.param["cost_input"],
        policy_function=request.param["policy_function"],
        policy_kwargs=request.param["policy_kwargs"],
        additional_mouselab_kwargs=additional_mouselab_kwargs,
    )

    trace = agent.simulate_trajectory()
    trace["pid"] = [0] * len(trace["states"])

    inference_policy_kwargs = deepcopy(request.param["policy_kwargs"])
    q_function_generator = (
        lambda cost_parameters, a, g: get_state_action_values(  # noqa : E731
            experiment_setting=request.param["env"]["setting"],
            cost_function=linear_depth,
            cost_parameters=cost_parameters,
            bmps_file="Myopic_VOC",
            structure=structure_dicts,
            env_params=env_params,
            bmps_path=Path(__file__).parents[0].joinpath("inputs/parameters/bmps/"),
            kappa=a,
            gamma=g,
        )
    )
    inference_policy_kwargs["q_function_generator"] = q_function_generator

    softmax_inference_agent_kwargs = {
        "participant_class": SymmetricMouselabParticipant,
        "participant_kwargs": {
            "experiment_setting": request.param["env"]["setting"],
            "policy_function": SoftmaxPolicy,
            "additional_mouselab_kwargs": additional_mouselab_kwargs,
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
            "additional_mouselab_kwargs": additional_mouselab_kwargs,
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
    traces, softmax_inference_agent_kwargs, _, _ = deepcopy(mle_test_cases)
    mle_algorithm = GridInference(traces, **softmax_inference_agent_kwargs)
    assert isinstance(mle_algorithm, GridInference)


def test_mle_run(mle_test_cases):
    (
        traces,
        softmax_inference_agent_kwargs,
        random_inference_agent_kwargs,
        correct_inference,
    ) = deepcopy(mle_test_cases)
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
    print("--------")
    for key, val in correct_inference.items():
        print(results.loc[results["map"].idxmax(), key], val)
        # assert results.loc[results["map"].idxmax(), key] == val
