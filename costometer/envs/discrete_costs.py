"""Provides cost functions for example discrete gym environments."""
import numpy as np
from scipy.spatial import distance


def distance_bonus(
    distance_cost_weight=None,
    positions_in_question=None,
    env_shape=None,
    distance_function=distance.euclidean,
):
    """
    Construct distance bonus cost function.

    :param distance_cost_weight: cost weight for being further from the end position
    :param positions_in_question: positions we bonus being close to (or far away from)
    :param env_shape: needed for getting position in these discrete environments
    :param distance_function: distance function to use
    :return: cost function to use
    """

    def cost_function(old_state, action, curr_state, done):
        if not done:
            curr_position = np.unravel_index(curr_state, env_shape)
            return (
                -min(
                    distance_function(position_in_question, curr_position)
                    for position_in_question in positions_in_question
                )
                * distance_cost_weight
            )
        else:
            return 0

    return cost_function


def potential_distance_bonus(
    distance_cost_weight=None,
    positions_in_question=None,
    env_shape=None,
    distance_function=distance.euclidean,
):
    """
    Construct distance bonus cost function.

    :param distance_cost_weight: cost weight for being further from the end position
    :param positions_in_question: positions we bonus being close to (or far away from)
    :param env_shape: needed for getting position in these discrete environments
    :param distance_function: distance function to use
    :return: cost function to use
    """

    def cost_function(old_state, action, curr_state, done):
        if not done:
            curr_position = np.unravel_index(curr_state, env_shape)
            old_position = np.unravel_index(old_state, env_shape)
            return (
                -min(
                    distance_function(position_in_question, curr_position)
                    for position_in_question in positions_in_question
                )
                * distance_cost_weight
                + min(
                    distance_function(position_in_question, old_position)
                    for position_in_question in positions_in_question
                )
                * distance_cost_weight
            )
        else:
            return 0

    return cost_function
