"""Provides functions for value iteration in discrete Open AI gym environments."""
from copy import deepcopy
from typing import Dict, Tuple, Union

from gym.envs.toy_text.discrete import DiscreteEnv

from ..envs.modified_mouselab import ModifiedMouseLabEnv


def flatten_q(
    outputted_q: Dict[Union[str, int], Dict[int, Union[int, float]]]
) -> Dict[Tuple[Union[str, int], int], Union[int, float]]:
    """
    Flatten Q so it is in dictionary format assumed by policy functions, e.g. Q[(s,a)] rather than Q[s][a].

    :param outputted_q: q as outputted by a planning algorithm like value iteration
    :return: flattened q
    """  # noqa: E501
    new_q = {}
    for state, state_values in outputted_q.items():
        for action, action_value in state_values.items():
            new_q[(state, action)] = action_value
    return new_q


def value_iteration(
    discrete_environment: Union[DiscreteEnv, ModifiedMouseLabEnv],
    gamma: float = 1.0,
    epsilon: float = 1e-6,
) -> Dict[Union[str, int], Dict[int, Union[int, float]]]:
    """
    Perform value iteration on discrete gym environment.

    :param discrete_environment: discrete gym environment
    :param gamma: discount factor
    :param epsilon: for testing convergence
    :return: Q, V, pi, info (empty)
    """
    V = {state: 0 for state in discrete_environment.P.keys()}
    Q = {state: {} for state in discrete_environment.P.keys()}
    policy_good_enough = False

    while not policy_good_enough:
        # save V for convergence check
        Vold = deepcopy(V)
        # innocent until proven guilty
        policy_good_enough = True

        # sweep through states and update V
        for state in discrete_environment.P.keys():
            for action in discrete_environment.actions(state):
                Q[state][action] = sum(
                    p * (r + gamma * V[s1])
                    for p, s1, r in discrete_environment.results(state, action)
                )
            V[state] = max(Q[state].values(), default=0)

        # check for convergence
        for state in Vold.keys():
            if abs(V[state] - Vold[state]) > epsilon:
                policy_good_enough = False
                break

        pi = {
            state: max(Q[state], key=Q[state].get)
            for state in V.keys()
            if state != discrete_environment.term_state
        }
    return Q, V, pi, {}
