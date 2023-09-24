"""Provides functions for marginalization and calculation of HDIs."""
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp


def normalize_maps(df, loglik_field):
    """
    Normalize MAP.

    :param df:
    :param loglik_field:
    :return:
    """
    df[f"{loglik_field}_normalized"] = log_softmax(df[loglik_field])
    return df


def marginalize_out_for_data_set(
    data: pd.DataFrame, cost_parameter_args: List[str], loglik_field: str = "map_test"
):
    """
    Marginalize out log likelihood for a dataframe.

    :param data:
    :param cost_parameter_args:
    :param loglik_field:
    :return:
    """
    marginal_probabilities = {parameter: [] for parameter in cost_parameter_args}
    sim_cols = [col for col in list(data) if "sim_" in col]
    for _, identifying_values in (
        data[["trace_pid"] + sim_cols].drop_duplicates().iterrows()
    ):
        curr_subset = deepcopy(
            data[
                data.apply(
                    lambda row: np.all(
                        [
                            row[col] == val
                            for col, val in zip(
                                identifying_values.index.values,  # noqa : B023
                                identifying_values.values,  # noqa : B023
                            )
                        ]
                    ),
                    axis=1,
                )
            ]
        )
        for parameter in cost_parameter_args:
            parameter_probabilities = marginalize_out_variables(
                curr_subset, loglik_field, parameter
            )
            marginal_probabilities[parameter].append(
                {
                    **dict(
                        zip(identifying_values.index.values, identifying_values.values)
                    ),
                    **parameter_probabilities,
                }
            )
    return marginal_probabilities


def marginalize_out_variables(
    df: pd.DataFrame, loglik_field: str, parameter: str
) -> Dict[Any, Any]:
    """
    Marginalize out log likelihood over a parameter.

    :param df:
    :param loglik_field:
    :param parameter:
    :return:
    """
    df = normalize_maps(df, loglik_field)

    marginalized_df = (
        df[[parameter, f"{loglik_field}_normalized"]]
        .groupby([parameter])
        .aggregate(logsumexp)
    )

    # normalize again (after groupby) so values are between 0 and 1
    marginalized_df[f"{loglik_field}_normalized"] = log_softmax(
        marginalized_df[f"{loglik_field}_normalized"]
    )
    # probabilities_sum_to_one = np.abs(logsumexp(df
    # [f"{loglik_field}_normalized"])-0) <= np.finfo(np.float64).eps
    # assert(probabilities_sum_to_one)

    # return parameter probability dict
    parameter_probabilities = marginalized_df.to_dict()[f"{loglik_field}_normalized"]

    return parameter_probabilities


def greedy_hdi_quantification(probs, vals):
    """
    Quantify HDI intervals greedily.

    :param probs:
    :param vals:
    :return:
    """
    include = np.zeros(len(vals))
    include[probs == np.amax(probs)] = 1

    if len(np.where(include == 1)[0]) > 1:
        edges = (np.where(include == 1)[0][0], np.where(include == 1)[0][-1])
        np.put(include, range(*edges), np.ones(len(range(*edges))))

    # greedily add until sums to .95
    possible_index = np.where(include == 0)[0]
    already_selected = np.where(include == 1)[0]
    neighbors = [
        el
        for selected in already_selected
        for el in [selected - 1, selected + 1]
        if el in possible_index
    ]
    while np.dot(include, probs) <= 0.95:
        max_neighbors = [
            neighbor
            for neighbor in neighbors
            if probs[neighbor] == np.amax(np.asarray(probs)[neighbors])
        ]
        include[max_neighbors] = 1

        if len(np.where(include == 1)[0]) > 1:
            edges = (np.where(include == 1)[0][0], np.where(include == 1)[0][-1])
            np.put(include, range(*edges), np.ones(len(range(*edges))))

        possible_index = np.where(include == 0)[0]
        already_selected = np.where(include == 1)[0]
        neighbors = [
            el
            for selected in already_selected
            for el in [selected - 1, selected + 1]
            if el in possible_index
        ]

    # possible we can remove a few from either side
    # e.g. for 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.81, 0.10, 0.02
    # include would currently be 0, 1, .... , 1 but 0, 1, ...., 1, 0 is best
    checked_removing = True
    while checked_removing:
        edges = [np.where(include == 1)[0][0], np.where(include == 1)[0][-1]]
        edge_probs = np.asarray(probs)[edges]
        min_edges = np.asarray(edges)[edge_probs == np.amin(edge_probs)]
        include[min_edges] = 0

        if np.dot(include, probs) < 0.95:
            checked_removing = False

    return list(np.asarray(vals)[edges])
