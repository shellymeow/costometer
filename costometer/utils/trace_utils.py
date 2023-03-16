"""These functions are used to transform human data to traces"""
from typing import List

import numpy as np
import pandas as pd
from mouselab.distributions import Categorical
from mouselab.env_utils import get_num_actions
from mouselab.envs.registry import registry
from mouselab.mouselab import MouselabEnv


def adjust_ground_truth(ground_truth, gamma, depth_view):
    return np.asarray(
        [
            round(
                ground_truth[node]
                * gamma ** (depth - 1),
                3,
            )
            if depth != 0
            else 0
            for node, depth in depth_view
        ]
    )


def adjust_state(state, gamma, depth_view, include_last_action=False):
    if state == '__term_state__':
        return state

    new_state = []
    for node, depth in depth_view:
        if depth == 0:
            # starting node
            new_state.append(node)
        elif hasattr(state[node], "sample"):
            vals = [
                round(val * gamma ** (depth - 1), 3)
                for val in state[node].vals
            ]
            new_state.append(Categorical(vals, state[node].probs))
        else:
            val = round(
                state[node]
                * gamma ** (depth - 1),
                3,
            )
            new_state.append(val)
    if include_last_action:
        new_state.append(state[-1])
    return tuple(new_state)


def get_row_property(row, column):
    if isinstance(row[column], str):
        return eval(row[column])
    else:
        return row[column]


def get_states_for_trace(
    actions, experiment_setting, ground_truth=None, **additional_mouselab_kwargs
):
    """
    Gets states, rewards and finished from actions
    :param actions:
    :param experiment_setting:
    :param ground_truth:
    :return:
    """
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting, ground_truth=ground_truth, **additional_mouselab_kwargs
    )
    state = env.reset()

    # finish initialize as -1, so if no action it will be -1
    resulting_trace = {"rewards": [], "states": [state], "finished": -1}

    for click in actions:
        state, reward, done, _ = env.step(int(click))
        resulting_trace["states"].append(state)
        resulting_trace["rewards"].append(reward)
        resulting_trace[
            "finished"
        ] = done  # on off-chance person quit half way we might have a False
    return resulting_trace


def get_trace_from_human_row(
    row, experiment_setting: str, include_last_action: bool = False
):
    """
    Transforms a human row to a trace
    :param row: row in mouselab dataframe
    :param experiment_setting: which (registered) mouselab setting is being used
    :param include_last_action:
    :return:
    """
    human_trace = {}

    # human only fields
    human_trace["movement_times"] = get_row_property(row, "action_times")
    human_trace["moves"] = get_row_property(row, "actions")
    human_trace["movement_rewards"] = get_row_property(row, "rewards")
    human_trace["path"] = get_row_property(row, "path")
    human_trace["block"] = row["block"]
    human_trace["trial_id"] = row["trial_id"]

    # ground truth for trial
    ground_truth_trial = get_row_property(row, "state_rewards")
    # depending on how mouselab file was processed starting node could be "" or dropped
    if ground_truth_trial[0] == "":
        ground_truth_trial[0] = 0  # initial node 0
    else:
        ground_truth_trial = [0] + ground_truth_trial
    human_trace["ground_truth"] = ground_truth_trial
    human_trace["trial_id"] = row["trial_id"]

    # not just actions, but also timing
    human_trace["queries"] = get_row_property(row, "queries")

    # fields in simulation traces (i_episode, states, actions rewards, finished, return)

    # episode number
    human_trace["i_episode"] = int(row["trial_index"])

    human_trace["actions"] = [
        int(click) for click in human_trace["queries"]["click"]["state"]["target"]
    ] + [
        get_num_actions(registry(experiment_setting).branching)
    ]  # add final action

    for trace_key, trace_val in get_states_for_trace(
        human_trace["actions"],
        experiment_setting,
        ground_truth=ground_truth_trial,
    ).items():
        human_trace[trace_key] = trace_val

    # return
    human_trace["return"] = sum(human_trace["rewards"]) + sum(
        human_trace["movement_rewards"]
    )

    # pid
    human_trace["pid"] = row["pid"]

    if include_last_action:
        human_trace["states"] = [
            (*state, last_action)
            for state, last_action in zip(
                human_trace["states"], [0] + human_trace["actions"][:-1]
            )
        ]

    return human_trace


def get_trajectories_from_participant_data(
    mouselab_mdp_dataframe,
    experiment_setting: str = "high_increasing",
    blocks: List[str] = None,
    include_last_action: bool = False,
):
    """
    Get trajectories for participants in a Mouselab MDP dataframe, given an experiment setting.

    :param mouselab_mdp_dataframe: Dataframe of participant mouselab-mdp trials
    :param experiment_setting:
    :param blocks:
    :param include_last_action:
    :return: Dictionary with same structure as a `trace` in mouselab-mdp
    """  # noqa: E501
    if blocks:
        mouselab_mdp_dataframe = mouselab_mdp_dataframe[
            mouselab_mdp_dataframe["block"].isin(blocks)
        ]

    # split dataframes into dataframe per subject
    mouselab_dict_traces = {
        pid: pid_df.apply(
            lambda row: get_trace_from_human_row(
                row, experiment_setting, include_last_action=include_last_action
            ),
            axis=1,
        ).values
        for pid, pid_df in mouselab_mdp_dataframe.groupby("pid")
    }

    mouselab_mdp_traces = [
        {key: [sub_dict[key] for sub_dict in dict_trace] for key in dict_trace[0]}
        for pid, dict_trace in mouselab_dict_traces.items()
    ]

    return mouselab_mdp_traces


def traces_to_df(traces):
    """

    :param traces:
    :return:
    """
    all_trace_dfs = []
    for trace in traces:
        # because s and a are lists of lists, we combine them
        #                          and then use pandas explode
        trace["s,a,r"] = []
        for i_episode in trace["i_episode"]:
            # remove final terminal state so s and a are both same size
            if "__term_state__" in trace["states"][i_episode]:
                trace["states"][i_episode].remove("__term_state__")
            trace["s,a,r"].append(
                list(
                    zip(
                        trace["states"][i_episode],
                        trace["actions"][i_episode],
                        trace["rewards"][i_episode],
                    )
                )
            )

        del trace["states"]
        del trace["actions"]
        del trace["rewards"]

        trace_df = pd.DataFrame.from_dict(trace)
        trace_df = trace_df.explode("s,a,r")
        # add states and actions back again, delete s,a
        trace_df["states"] = trace_df["s,a,r"].apply(lambda sa: sa[0])
        trace_df["actions"] = trace_df["s,a,r"].apply(lambda sa: sa[1])
        trace_df["rewards"] = trace_df["s,a,r"].apply(lambda sa: sa[2])
        del trace_df["s,a,r"]

        all_trace_dfs.append(trace_df)

    return pd.concat(all_trace_dfs)
