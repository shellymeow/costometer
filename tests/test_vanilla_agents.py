from copy import deepcopy
from pathlib import Path

import dill as pickle
import numpy as np
import pytest
from .inputs.create_test_inputs import create_trajectory, get_q_function
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from mouselab.policies import SoftmaxPolicy

from costometer.agents.vanilla import Participant, SymmetricMouselabParticipant
from costometer.envs.discrete import (
    ModifiedCliffWalkingEnv,
    ModifiedVerySimpleGridWorld,
)
from costometer.envs.discrete_costs import distance_bonus
from costometer.planning_algorithms.vi import flatten_q, value_iteration

# HACK for compatibility with gym=0.21.0 and numpy>=1.20 (np.bool deprecated)
np.bool = bool

vanilla_agent_test_data = [
    {
        "env": {
            "name": "small_decreasing",
            "branching": [2, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_decreasing_reward,
        },
        "additional_settings": {"num_trials": 5, "policy_function": SoftmaxPolicy},
    },
    {
        "env": {
            "name": "small_increasing",
            "branching": [2, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        },
        "additional_settings": {"num_trials": 10, "policy_function": SoftmaxPolicy},
    },
]


@pytest.fixture(params=vanilla_agent_test_data)
def vanilla_agent_test_cases(request):
    register(**request.param["env"])

    # generate test q values or trajectories if they don't already exist
    if (
        not Path(__file__)
        .parents[0]
        .joinpath(f"inputs/{request.param['env']['name']}_trace.pickle")
        .is_file()
    ):
        create_trajectory(request.param["env"]["name"])
    if (
        not Path(__file__)
        .parents[0]
        .joinpath(f"inputs/{request.param['env']['name']}_q_function.pickle")
        .is_file()
    ):
        get_q_function(request.param["env"]["name"])

    yield request.param["env"]["name"], request.param["additional_settings"]


def test_instantiate_symmetric_participant(vanilla_agent_test_cases):
    setting, additional_settings = vanilla_agent_test_cases
    with open(
        Path(__file__).parents[0].joinpath(f"inputs/{setting}_q_function.pickle"), "rb"
    ) as q_file:
        q_function = pickle.load(q_file)

    participant = SymmetricMouselabParticipant(
        setting, policy_kwargs={"preference": q_function}, **additional_settings
    )
    assert isinstance(participant, SymmetricMouselabParticipant)


def test_deepcopy_symmetric_participant(vanilla_agent_test_cases):
    setting, additional_settings = vanilla_agent_test_cases
    with open(
        Path(__file__).parents[0].joinpath(f"inputs/{setting}_q_function.pickle"), "rb"
    ) as q_file:
        q_function = pickle.load(q_file)

    participant = SymmetricMouselabParticipant(
        setting, policy_kwargs={"preference": q_function}, **additional_settings
    )
    assert isinstance(deepcopy(participant), SymmetricMouselabParticipant)
    assert deepcopy(participant) != participant


def test_simulate_trajectory(vanilla_agent_test_cases):
    setting, additional_settings = vanilla_agent_test_cases
    with open(
        Path(__file__).parents[0].joinpath(f"inputs/{setting}_q_function.pickle"), "rb"
    ) as q_file:
        q_function = pickle.load(q_file)

    participant = SymmetricMouselabParticipant(
        setting, policy_kwargs={"preference": q_function}, **additional_settings
    )
    participant.simulate_trajectory()

    with open(
        Path(__file__).parents[0].joinpath(f"inputs/{setting}_trace.pickle"), "rb"
    ) as trace_file:
        trace = pickle.load(trace_file)

    # participant trajectory will have ground truth and trial id in it
    assert participant.trace.keys() - {"ground_truth", "trial_id"} == trace.keys()


def test_infer_trajectory(vanilla_agent_test_cases):
    setting, additional_settings = vanilla_agent_test_cases
    with open(
        Path(__file__).parents[0].joinpath(f"inputs/{setting}_q_function.pickle"), "rb"
    ) as q_file:
        q_function = pickle.load(q_file)

    participant = SymmetricMouselabParticipant(
        setting, policy_kwargs={"preference": q_function}, **additional_settings
    )
    participant.simulate_trajectory()
    logliks = participant.compute_likelihood(participant.trace)

    probabilities = np.exp(np.concatenate(logliks))

    # check probabilities all between 0 and 1
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # check shape matches
    assert [len(trial) for trial in logliks] == [
        len(actions) for actions in participant.trace["actions"]
    ]


discrete_env_test_data = [
    [
        ModifiedCliffWalkingEnv,
        5,
        distance_bonus,
        {
            "distance_cost_weight": -0.05,
            "positions_in_question": list(
                zip(*np.where(ModifiedCliffWalkingEnv()._cliff))
            ),
            "env_shape": ModifiedCliffWalkingEnv().shape,
        },
    ],
    [
        ModifiedVerySimpleGridWorld,
        3,
        distance_bonus,
        {
            "distance_cost_weight": -0.2,
            "positions_in_question": [(0, 0)],
            "env_shape": ModifiedVerySimpleGridWorld().shape,
        },
    ],
]


@pytest.fixture(params=discrete_env_test_data)
def discrete_env_test_cases(request):
    yield request.param


def test_discrete_participant_instantiation(
    discrete_env_test_cases,
):
    discrete_class, num_trials, cost_function, cost_kwargs = discrete_env_test_cases
    envs = [
        discrete_class(cost_function=cost_function, cost_kwargs=cost_kwargs)
        for _ in range(num_trials)
    ]
    Q, _, _, _ = value_iteration(envs[0])

    participant = Participant(
        envs=envs,
        num_trials=1,
        cost_function=None,
        cost_kwargs={},
        policy_function=SoftmaxPolicy,
        policy_kwargs={"preference": flatten_q(Q)},
    )
    assert isinstance(participant, Participant)


def test_discrete_simulate_and_infer(discrete_env_test_cases):
    discrete_class, num_trials, cost_function, cost_kwargs = discrete_env_test_cases
    envs = [
        discrete_class(cost_function=cost_function, cost_kwargs=cost_kwargs)
        for _ in range(num_trials)
    ]
    Q, _, _, _ = value_iteration(envs[0])
    participant = Participant(
        envs=envs,
        num_trials=1,
        cost_function=None,
        cost_kwargs={},
        policy_function=SoftmaxPolicy,
        policy_kwargs={"preference": flatten_q(Q)},
    )
    participant.simulate_trajectory()
    logliks = participant.compute_likelihood(participant.trace)

    probabilities = np.exp(np.concatenate(logliks))

    # check probabilities all between 0 and 1
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # check shape matches
    assert [len(trial) for trial in logliks] == [
        len(actions) for actions in participant.trace["actions"]
    ]
