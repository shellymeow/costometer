"""Provides utility functions to calculate bias (or type of click) calculations."""
from copy import deepcopy
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from mouselab.graph_utils import graph_from_adjacency_list
from mouselab.mouselab import MouselabEnv


def get_possible_actions_for_state(state):
    """
    Get possible actions for state.

    :param state: state from MouseLab gym environment
    :return:
    """
    possible_actions = [
        action_idx
        for action_idx, action in enumerate(state)
        if hasattr(action, "sample")
    ] + [len(state)]
    return possible_actions


def get_curr_optimal_for_state(state, q_dictionary: Dict[Tuple[Any, Any], float]):
    """
    Get optimal actions, for a state a participant finds themselves in.

    :param state: state (as in MouselabEnv class)
    :param q_dictionary: loaded q dictionary
    :return: list of optimal actions
    """
    possible_actions = [
        action_idx
        for action_idx, action in enumerate(state)
        if hasattr(action, "sample")
    ] + [len(state)]
    action_values = {
        action: q_dictionary[(state, action)] for action in possible_actions
    }
    curr_optimal = [
        action
        for action, value in action_values.items()
        if value == max(action_values.values())
    ]
    return curr_optimal


def get_optimal_trajectories_for_env(env, q_dictionary):
    """
    Get optimal trajectories for environment (given Q value dictionary).

    :param env:
    :param q_dictionary:
    :return:
    """
    terminal_action = env.term_action

    # initialize exploration set and list to save trajectories in
    current_exploration_set = []
    explored_set = []

    for action in get_curr_optimal_for_state(env._state, q_dictionary):
        curr_mouse = deepcopy(env)
        if action == terminal_action:
            # assign reward since last action
            final_mouse = deepcopy(curr_mouse)
            curr_state, rew, _, _ = final_mouse.step(action)
            explored_set.append([[curr_mouse, action, rew]])
        else:
            current_exploration_set.append([[curr_mouse, action, np.nan]])

    # loop through until all optimal trajectories have ended
    while len(current_exploration_set) > 0:
        curr_trajectory = current_exploration_set.pop(0)
        last_sar_triple = curr_trajectory[-1]
        curr_mouse = deepcopy(last_sar_triple[0])
        curr_state, rew, _, _ = curr_mouse.step(last_sar_triple[1])
        # assign reward for last action
        last_sar_triple[-1] = rew

        for next_action in get_curr_optimal_for_state(curr_state, q_dictionary):
            if next_action == terminal_action:
                # assign reward since last action
                final_mouse = deepcopy(curr_mouse)
                curr_state, rew, _, _ = final_mouse.step(next_action)
                explored_set.append(curr_trajectory + [[curr_mouse, next_action, rew]])
            else:
                current_exploration_set.append(
                    curr_trajectory + [[curr_mouse, next_action, np.nan]]
                )

    optimal_trajectories = []
    for trajectory in explored_set:
        optimal_trajectories.append(
            [(state._state, action, reward) for state, action, reward in trajectory]
        )

    return optimal_trajectories


def get_possible_paths(env):
    """
    Get possible paths for environment.

    This might do the same thing as the optimal_paths method in the mouselab toolbox
    (but that had no documentation and was not as straightforward)
    """
    G = graph_from_adjacency_list(env.tree)
    final_nodes = [x for x in G.nodes() if G.out_degree(x) == 0]
    # perhaps in some environments there will be multiple starting nodes
    start_nodes = [x for x in G.nodes() if G.in_degree(x) == 0]
    possible_paths = []
    for start_node in start_nodes:
        # starting nodes are not included in path (usually spider or plane sits on them)
        possible_paths.extend(
            [
                possible_path[1:]
                for possible_path in nx.algorithms.simple_paths.all_simple_paths(
                    G, start_node, final_nodes
                )
            ]
        )
    return possible_paths


def get_max_unbounded_vals_for_trial(ground_truth, possible_paths):
    """
    "Unbounded" value is the path people would take if they could inspect all nodes.

    :param ground_truth:
    :param possible_paths:
    :return:
    """
    ground_truth = np.asarray(ground_truth)
    possible_rewards = [ground_truth[possible_path] for possible_path in possible_paths]

    max_sum = max([sum(possible_reward) for possible_reward in possible_rewards])
    max_first = max(
        [
            possible_reward[0]
            for possible_reward in possible_rewards
            if sum(possible_reward) == max_sum
        ]
    )
    return max_first, max_sum


def get_max_unbounded_vals_for_trials(ground_truth_dicts, possible_paths):
    """
    Get max unbounded values for multiple trials, see get_max_unbounded_vals_for_trial for more info.

    :param ground_truth_dicts:
    :param possible_paths:
    :return:
    """  # noqa: E501
    max_early_node_for_trials = {}
    max_sum_for_trials = {}
    for trial_id, ground_truth in ground_truth_dicts.items():
        max_first, max_sum = get_max_unbounded_vals_for_trial(
            ground_truth, possible_paths
        )
        max_early_node_for_trials[trial_id] = max_first
        max_sum_for_trials[trial_id] = max_sum
    return max_early_node_for_trials, max_sum_for_trials


def add_unbounded_bias(
    mouselab_df: pd.DataFrame, env: MouselabEnv, ground_truth_dicts: Dict[Any, Any]
):
    """
    Add unbounded bias to mouselab dataframe.

    :param mouselab_df:
    :param env:
    :param ground_truth_dicts:
    :return:
    """
    possible_paths = get_possible_paths(env)
    max_early_node_for_trials, max_sum_for_trials = get_max_unbounded_vals_for_trials(
        ground_truth_dicts, possible_paths
    )
    mouselab_df["unbounded_present_bias"] = mouselab_df.apply(
        lambda row: row["ground_truth"][row["taken_paths"][0]]
        - max_early_node_for_trials[row["trial_id"]],
        axis=1,
    )
    mouselab_df["unbounded_loss"] = mouselab_df.apply(
        lambda row: sum(np.asarray(row["ground_truth"])[row["taken_paths"]])
        - max_sum_for_trials[row["trial_id"]],
        axis=1,
    )
    return mouselab_df


def get_optimal_trajectories_for_ground_truths(
    ground_truths_dict,
    q_dictionary,
    cost_function,
    cost_parameters,
    experiment_setting="high_increasing",
):
    """
    Get optimal trajectories for specific trials ("ground truths").

    :param ground_truths_dict:
    :param q_dictionary:
    :param cost_function:
    :param experiment_setting:
    :return:
    """
    optimal_trajectories = {}
    for trial_num, ground_truth in ground_truths_dict.items():
        env = MouselabEnv.new_symmetric_registered(
            experiment_setting,
            ground_truth=ground_truth,
            cost=cost_function(**cost_parameters),
        )
        curr_trajectories = get_optimal_trajectories_for_env(env, q_dictionary)
        optimal_trajectories[trial_num] = curr_trajectories
    return optimal_trajectories


def get_optimal_trajectories_suppl_dicts(optimal_trajectories, node_classification):
    """
    Get optimal trajectories.

    :param optimal_trajectories:
    :param node_classification:
    :return:
    """
    suppl_dicts = {f"num_{key}": {} for key in node_classification.keys()}
    for trial_id, ground_truth_traj in optimal_trajectories.items():
        for key in node_classification.keys():
            suppl_dicts[f"num_{key}"][trial_id] = [
                len(set([a for s, a, r in traj]).intersection(node_classification[key]))
                for traj in ground_truth_traj
            ]
    return suppl_dicts


def add_bounded_bias(
    processed_mouselab,
    ground_truths_dict,
    q_dictionary,
    node_classification,
    cost_function,
    cost_parameters,
):
    optimal_trajectories = get_optimal_trajectories_for_ground_truths(
        ground_truths_dict, q_dictionary, cost_function, cost_parameters
    )
    ratios = get_optimal_trajectories_suppl_dicts(
        optimal_trajectories, node_classification
    )

    processed_mouselab["over_under"] = processed_mouselab.apply(
        lambda row: calculate_overplanning(row, ratios), axis=1
    )
    processed_mouselab["present"] = processed_mouselab.apply(
        lambda row: calculate_present_bias(row, ratios),
        axis=1,
    )
    processed_mouselab["present_late"] = processed_mouselab.apply(
        lambda row: calculate_present_bias_late(row, ratios),
        axis=1,
    )
    return processed_mouselab


def get_subset_node_ratio(num_subset, num_clicks):
    if num_clicks == 0:
        return 0
    else:
        return num_subset / num_clicks


def calculate_overplanning(
    row,
    suppl_dicts,
    num_clicks_field="num_clicks",
    num_simulated_clicks_field="num_clicks",
):
    return row[num_clicks_field] - np.mean(
        suppl_dicts[num_simulated_clicks_field][row["trial_id"]]
    )


def calculate_present_bias(
    row,
    suppl_dicts,
    num_early_field="num_early",
    ratio_simulated_early_field="num_early",
    num_clicks_field="num_clicks",
):
    return get_subset_node_ratio(row[num_early_field], row[num_clicks_field]) - np.mean(
        suppl_dicts[ratio_simulated_early_field][row["trial_id"]]
    )


def calculate_present_bias_late(
    row,
    suppl_dicts,
    num_late_field="num_late",
    ratio_simulated_late_field="num_late",
    num_clicks_field="num_clicks",
):
    return (1 - get_subset_node_ratio(row[num_late_field], row[num_clicks_field])) - (
        1 - np.mean(suppl_dicts[ratio_simulated_late_field][row["trial_id"]])
    )


def get_trial_loss(row, q_dictionary, final_action=None):
    possible_actions = get_possible_actions_for_state(row["state"])
    max_q = max(
        q_dictionary[(row["state"], possible_action)]
        for possible_action in possible_actions
    )

    if final_action:
        action = final_action if row["actions"] == 0 else row["actions"]
    else:
        action = row["actions"]

    loss = q_dictionary[(row["state"], action)] - max_q
    return loss


def add_states_simulated_row(row, experiment_setting="high_increasing"):
    # doesn't depend on cost
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting, ground_truth=row["ground_truth"]
    )

    before_current_action = True
    curr_action_idx = 0
    while before_current_action:
        curr_action = row["full_actions"][curr_action_idx]
        if curr_action == row["actions"]:
            before_current_action = False
        else:
            env.step(curr_action)

            curr_action_idx += 1
    return env._state


def add_processed_columns(
    mouselab_datas,
    experiment_setting,
    ground_truths_dict,
    node_classification,
    cost_function,
    cost_parameters,
    human=False,
    q_dictionary=None,
):
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting, cost=cost_function(**cost_parameters)
    )

    mouselab_datas = add_unbounded_bias(mouselab_datas, env, ground_truths_dict)
    if q_dictionary is not None:
        mouselab_datas = add_bounded_bias(
            mouselab_datas,
            ground_truths_dict,
            q_dictionary,
            node_classification,
            cost_function,
            cost_parameters,
        )
    if human:
        raise NotImplementedError
    else:
        mouselab_datas["state"] = mouselab_datas.apply(
            lambda row: add_states_simulated_row(row, experiment_setting), axis=1
        )
        if q_dictionary is not None:
            mouselab_datas["loss"] = mouselab_datas.apply(
                lambda row: get_trial_loss(
                    row, q_dictionary, final_action=env.term_action
                ),
                axis=1,
            )

    return mouselab_datas


def add_click_count_columns_to_simulated(simulated_mouselab_df, click_type_dict):
    """
    Add click count columns to simulated data dataframe.

    :param simulated_mouselab_df:
    :param click_type_dict:
    :return:
    """
    for click_type, clicks in click_type_dict.items():
        simulated_mouselab_df[f"num_{click_type}"] = simulated_mouselab_df[
            "actions"
        ].apply(
            lambda query: query in clicks  # noqa : B023
        )
    return simulated_mouselab_df


def fix_trial_id_for_simulated(simulated_mouselab_df, reward_json):
    """
    Fix trial ID, given original reward file.

    :param simulated_mouselab_df: mouselab dataframe with state_rewards column
    :param reward_json: json output of original file
    :return: modified mouselab dataframe with trial_id column
    """
    data = {}

    # create mapping of ground truth rewards to trial index
    for reward in reward_json:
        string_reward = [str(int(entry)) for entry in reward["stateRewards"]]
        data["".join(string_reward)] = reward["trial_id"]

    simulated_mouselab_df["trial_id"] = simulated_mouselab_df["ground_truth"].apply(
        lambda state_rewards: data[
            "".join([str(int(entry)) for entry in state_rewards])
        ]
    )
    return simulated_mouselab_df
