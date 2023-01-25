"""Utility functions for MAP calculation, priors and finding the best parameters."""
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Union

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from more_itertools import powerset
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from scipy import stats  # noqa
from scipy.stats import rv_continuous
from statsmodels.tools.eval_measures import bic

from costometer.agents import SymmetricMouselabParticipant
from costometer.utils.cost_utils import get_param_string, load_q_file
from costometer.utils.plotting_utils import generate_model_palette
from costometer.utils.trace_utils import (
    get_trajectories_from_participant_data,
    traces_to_df,
)


def get_best_parameters(
    df: pd.DataFrame,
    cost_details: Dict[str, Any],
    priors: Dict[Any, Any],
):
    """
    Get best parameters for a dataframe, for certain cost_details

    :param df:
    :param cost_details:
    :param priors:
    :return:
    """
    best_parameter_values = {}

    # reset index's df for the indexing by best row
    df = df.reset_index()

    for prior_type, prior_dict in priors.items():
        # save best parameters for each prior
        best_parameter_values[prior_type] = {}
        for subset in powerset(cost_details["constant_values"]):
            # subset dataframe
            curr_data = df[
                df.apply(
                    lambda row: sum(
                        row[cost_param] == cost_details["constant_values"][cost_param]
                        for cost_param in list(subset)
                    )
                    == len(list(subset)),
                    axis=1,
                )
            ]

            # add prior
            curr_data[f"map_{prior_type}"] = curr_data.apply(
                lambda row: row["mle"]
                + sum(
                    [
                        np.log(prior_dict[param][row[param]])
                        for param in prior_dict.keys()
                        if param not in subset
                    ]
                ),
                axis=1,
            )

            # when multiple pids included,
            # some might be duplicated (e.g. pid 0 with sim cost 1 vs 2)
            sim_cols = [col for col in list(curr_data) if "sim_" in col]

            best_param_rows = df.loc[  #
                curr_data.groupby(["trace_pid"] + sim_cols).idxmax()[
                    f"map_{prior_type}"
                ]
            ]
            assert np.all(
                [
                    counter == 1
                    for pid, counter in Counter(
                        best_param_rows[["trace_pid"] + sim_cols]
                        .to_records(index=False)
                        .tolist()
                    ).most_common()
                ]
            )

            best_parameter_values[prior_type][subset] = best_param_rows

    return best_parameter_values


def add_cost_priors_to_temp_priors(
    softmax_df: pd.DataFrame,
    cost_details: Dict[str, Any],
    temp_prior_details: Dict[str, Any],
    additional_params=List[str],
) -> Dict[str, Dict[str, Any]]:
    """

    :param softmax_df:
    :param cost_details:
    :param temp_prior_details:
    :param additional_params:
    :return:
    """
    full_priors = {}
    for prior, prior_inputs in temp_prior_details.items():
        priors = {}

        temp_prior = get_temp_prior(
            rv=eval(prior_inputs["rv"]),
            possible_vals=prior_inputs["possible_temps"],
            inverse=prior_inputs["inverse"],
        )
        priors["temp"] = dict(zip(temp_prior.vals, temp_prior.probs))
        for cost_parameter_arg in (
            list(cost_details["constant_values"]) + additional_params
        ):
            numeric_values = softmax_df[cost_parameter_arg][
                softmax_df[cost_parameter_arg].apply(
                    lambda entry: not isinstance(entry, str)
                )
            ]
            unique_args = np.unique(numeric_values)
            priors[cost_parameter_arg] = dict(
                zip(unique_args, np.ones(len(unique_args)) * 1 / len(unique_args))
            )

        assert np.all(
            [np.sum(priors[param_key].values()) for param_key in priors.keys()]
        )
        full_priors[prior] = priors
    return full_priors


def extract_mles_and_maps(
    data: pd.DataFrame,
    cost_details: Dict[str, Any],
    priors: Dict[Any, Any],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """

    :param data:
    :param cost_details:
    :params priors:
    :return:
    """
    # save random data first
    best_parameter_values = {}

    random_data = data[data["applied_policy"] == "RandomPolicy"].reset_index(drop=True)
    for parameter_arg in priors["uniform"].keys():
        random_data[parameter_arg] = np.nan
    best_parameter_values["RandomPolicy"] = random_data

    # now only consider softmax policy
    softmax_data = data[data["applied_policy"] == "SoftmaxPolicy"].reset_index(
        drop=True
    )

    best_parameter_values = {
        **best_parameter_values,
        "SoftmaxPolicy": get_best_parameters(softmax_data, cost_details, priors),
    }
    return best_parameter_values


def get_temp_prior(
    rv: rv_continuous, possible_vals: List[float], inverse: bool = True
) -> Categorical:
    """

    :param rv:
    :param possible_vals:
    :param inverse:
    :return:
    """
    if inverse:
        rv_vals = [1 / val for val in possible_vals]
    else:
        rv_vals = possible_vals

    normalizing_factor = sum([rv.pdf(val) for val in rv_vals])
    categorical_dist = Categorical(
        possible_vals, [rv.pdf(val) / normalizing_factor for val in rv_vals]
    )
    return categorical_dist


def find_best_parameters(fitting_data_df, function="min", objective="loss"):
    """

    :param fitting_data_df:
    :param function:
    :param objective:
    :return:
    """
    sim_cols = [col for col in list(fitting_data_df) if "sim_" in col]

    best_values = (
        fitting_data_df.groupby(["trace_pid"] + sim_cols)
        .aggregate(function)[objective]
        .to_dict()
    )

    if len(sim_cols) > 0:
        best_param_rows = fitting_data_df[
            fitting_data_df.apply(
                lambda row: best_values[
                    tuple([row[sim_col] for sim_col in ["trace_pid"] + sim_cols])
                ]
                == row["loss"],
                axis=1,
            )
        ]

    else:
        best_param_rows = fitting_data_df[
            fitting_data_df.apply(
                lambda row: best_values[row["trace_pid"]] == row["loss"], axis=1
            )
        ]

    return best_param_rows


class AnalysisObject:
    def __init__(
        self,
        experiment_name: str,
        irl_path: Union[str, Path],
        experiment_subdirectory: Union[str, Path],
    ):
        """

        :param experiment_name:
        :param irl_path:
        """
        self.experiment_name = experiment_name

        #  must match my folder structure
        #  subfolders data/processed/{experiment_name}
        #  & analysis/{experiment_subdirectory} should exist
        self.irl_path = irl_path
        self.experiment_subdirectory = experiment_subdirectory

        # add yaml attributes to object, should be:
        # sessions, cost_functions,
        self.read_experiment_yaml()
        self.load_cost_function_details()

        if not hasattr(self, "palette_name"):
            self.palette_name = experiment_name

        if not hasattr(self, "q_path"):
            self.q_path = None

        if not self.simulated:
            individual_variables = [
                pd.read_csv(
                    self.irl_path.joinpath(
                        f"data/processed/{session}/individual-variables.csv"
                    )
                )
                for session in self.sessions
                if self.irl_path.joinpath(
                    f"data/processed/{session}/individual-variables.csv"
                ).is_file()
            ]
            self.individual_variables = (
                pd.concat(individual_variables)
                if len(individual_variables) > 0
                else None
            )

            self.mouselab_trials = pd.concat(
                [
                    pd.read_csv(
                        self.irl_path.joinpath(
                            f"data/processed/{session}/mouselab-mdp.csv"
                        )
                    )
                    for session in self.sessions
                ]
            )

            quiz_and_demos = [
                pd.read_csv(
                    self.irl_path.joinpath(
                        f"data/processed/{session}/quiz-and-demo.csv"
                    )
                )
                for session in self.sessions
                if self.irl_path.joinpath(
                    f"data/processed/{session}/quiz-and-demo.csv"
                ).is_file()
            ]

            self.quest = pd.concat(quiz_and_demos) if len(quiz_and_demos) > 0 else None

            self.load_session_details()
            if not hasattr(self, "params"):
                self.params = [""]
        else:
            self.session_details = {
                session: {
                    "experiment_setting": session.split("/")[1],
                    "trials_per_block": None,
                }
                for session in self.sessions
            }
            self.block = ["All"]

            # currently only implemented for one cost function
            assert len(self.cost_functions) == 1

            if not self.params:
                cost_strings = [
                    get_param_string(
                        dict(
                            zip(
                                self.cost_details[self.cost_functions[0]][
                                    "cost_parameter_args"
                                ],
                                prod,
                            )
                        )
                    )
                    for prod in product(
                        *[
                            self.__getattribute__(param)
                            for param in self.cost_details[self.cost_functions[0]][
                                "cost_parameter_args"
                            ]
                        ]
                    )
                ]
                self.params = [
                    f"_{cost_string}_{temp:.2f}"
                    for temp, cost_string in product(self.temp, cost_strings)
                ]

        # only programmed correctly if all sessions have same experiment setting
        assert (
            len(
                np.unique(
                    [
                        session_details["experiment_setting"]
                        for session_details in self.session_details.values()
                    ]
                )
            )
            == 1
        )
        self.experiment_setting = [
            session_details["experiment_setting"]
            for session_details in self.session_details.values()
        ][0]

        self.optimization_data = self.load_optimization_data()

        if not self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/data/"
            f"{self.experiment_name}_models_palette.pickle"
        ).is_file():
            static_palette = generate_model_palette(
                self.optimization_data["Model Name"].unique()
            )
            self.irl_path.joinpath(
                f"analysis/{self.experiment_subdirectory}/data/"
            ).mkdir(parents=True, exist_ok=True)
            with open(
                self.irl_path.joinpath(
                    f"analysis/{self.experiment_subdirectory}/data/"
                    f"{self.palette_name}_models_palette.pickle"
                ),
                "wb",
            ) as f:
                pickle.dump(static_palette, f)

    def load_cost_function_details(self):
        self.cost_details = {}
        for cost_function in self.cost_functions:
            yaml_file = self.irl_path.joinpath(
                f"data/inputs/yamls/cost_functions/{cost_function}.yaml"
            )
            with open(str(yaml_file), "r") as stream:
                self.cost_details[cost_function] = yaml.safe_load(stream)

    def load_session_details(self):
        self.session_details = {}
        for session in self.sessions:
            yaml_file = self.irl_path.joinpath(
                f"data/inputs/yamls/experiments/{session}.yaml"
            )
            with open(str(yaml_file), "r") as stream:
                self.session_details[session] = yaml.safe_load(stream)

    def load_optimization_data(self):
        full_dfs = []
        for cost_function in self.cost_functions:
            cost_dfs = []
            for session in self.sessions:
                for param in self.params:
                    mle_and_map_files = list(
                        self.irl_path.glob(
                            f"data/processed/{session}{param}/{cost_function}"
                            f"/mle_and_map*.pickle"
                        )
                    )
                    for mle_and_map_file in mle_and_map_files:
                        with open(
                            mle_and_map_file,
                            "rb",
                        ) as f:
                            data = pickle.load(f)

                        # remove possibility with map without prior
                        if "map" in data["SoftmaxPolicy"]:
                            del data["SoftmaxPolicy"]["map"]
                        if "map" in data["RandomPolicy"]:
                            del data["RandomPolicy"]["map"]
                        if "map" in data["Group"]:
                            del data["Group"]["map"]

                        random_df = data["RandomPolicy"]
                        random_df["Model Name"] = "Null"
                        random_df["Number Parameters"] = 0

                        # make long
                        map_and_mle_cols = [
                            col for col in list(random_df) if "mle" in col
                        ]
                        cols = [
                            col
                            for col in list(random_df)
                            if col not in map_and_mle_cols
                        ]
                        random_df = random_df.melt(
                            id_vars=cols, value_vars=map_and_mle_cols, var_name="metric"
                        )

                        if not self.simulated:
                            random_df["Block"] = random_df["metric"].apply(
                                lambda metric: "_".join(metric.split("_")[:-2])
                                if "_".join(metric.split("_")[:-2])
                                in self.session_details[session]["trials_per_block"]
                                else "None"
                            )
                            random_df["Number Trials"] = random_df["Block"].apply(
                                lambda block: self.session_details[session][
                                    "trials_per_block"
                                ][block]
                                if block
                                in self.session_details[session]["trials_per_block"]
                                else sum(
                                    self.session_details[session][
                                        "trials_per_block"
                                    ].values()
                                )
                            )
                        else:
                            random_df["Block"] = "All"
                            random_df["Number Trials"] = self.number_trials

                        random_df["Prior"] = "None"
                        random_df["Group"] = False
                        random_df["session"] = param
                        cost_dfs.append(random_df)

                        for softmax_type, group in zip(
                            ["Group", "SoftmaxPolicy"], [True, False]
                        ):
                            for metric in data[softmax_type].keys():
                                metric_dfs = []
                                for removed_params in data[softmax_type][metric].keys():
                                    if removed_params in eval(
                                        self.cost_details[cost_function]["model_name"]
                                    ):
                                        curr_df = data[softmax_type][metric][
                                            removed_params
                                        ].copy(deep=True)
                                        curr_df["Model Name"] = eval(
                                            self.cost_details[cost_function][
                                                "model_name"
                                            ]
                                        )[removed_params]
                                        curr_df["Number Parameters"] = (
                                            len(
                                                self.cost_details[cost_function][
                                                    "constant_values"
                                                ]
                                            )
                                            + 1
                                            - len(removed_params)
                                        )

                                        metric_dfs.append(curr_df)
                                metric_df = pd.concat(metric_dfs).reset_index(drop=True)
                                metric_df.drop(
                                    [
                                        col
                                        for col in list(metric_df)
                                        if ("map" in col or "mle" in col)
                                        and (col != metric)
                                    ],
                                    axis=1,
                                    inplace=True,
                                )
                                metric_df.rename(
                                    columns={metric: "value"}, inplace=True
                                )
                                metric_df["metric"] = metric

                                if not self.simulated:
                                    metric_df["Block"] = metric_df["metric"].apply(
                                        lambda metric: "_".join(metric.split("_")[:-2])
                                        if "_".join(metric.split("_")[:-2])
                                        in self.session_details[session][
                                            "trials_per_block"
                                        ]
                                        else "None"
                                    )
                                    metric_df["Number Trials"] = metric_df[
                                        "Block"
                                    ].apply(
                                        lambda block: self.session_details[session][
                                            "trials_per_block"
                                        ][block]
                                        if block
                                        in self.session_details[session][
                                            "trials_per_block"
                                        ]
                                        else sum(
                                            self.session_details[session][
                                                "trials_per_block"
                                            ].values()
                                        )
                                    )
                                else:
                                    metric_df["Block"] = "All"
                                    metric_df["Number Trials"] = self.number_trials

                                metric_df["Prior"] = metric_df["metric"].apply(
                                    lambda metric: metric.split("_")[-1]
                                    if metric.split("_")[-1] not in ["mle", "map"]
                                    else "None"
                                )
                                metric_df["Group"] = group
                                metric_df["session"] = param
                                cost_dfs.append(metric_df)

            cost_df = pd.concat(cost_dfs)
            cost_df["cost_function"] = cost_function
            full_dfs.append(cost_df)
        full_df = pd.concat(full_dfs).reset_index(drop=True)

        full_df["bic"] = full_df.apply(
            lambda row: bic(
                llf=row["value"],
                nobs=row["Number Trials"],
                df_modelwc=row["Number Parameters"],
            ),
            axis=1,
        )
        return full_df

    def read_experiment_yaml(self):
        """

        :return:
        """
        yaml_file = self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/"
            f"inputs/yamls/{self.experiment_name}.yaml"
        )
        with open(str(yaml_file), "r") as stream:
            yaml_dict = yaml.safe_load(stream)
        # append all entries in yaml_dict as attributes
        for key in yaml_dict:
            setattr(self, key, yaml_dict[key])

    def add_individual_variables(
        self, df: pd.DataFrame, variables_of_interest: List[str] = None
    ) -> pd.DataFrame:
        return df.merge(
            self.individual_variables[["pid", *variables_of_interest]],
            left_on=[
                "trace_pid",
            ],
            right_on=["pid"],
            how="left",
        )

    def add_mouselab_columns(
        self, df: pd.DataFrame, variables_of_interest: List[str] = None
    ) -> pd.DataFrame:
        if variables_of_interest is None:
            variables_of_interest = ["block"]

        return df.merge(
            self.mouselab_trials[["pid", *variables_of_interest]],
            left_on=["trace_pid"],
            right_on=["pid"],
            how="left",
        )

    def query_optimization_data(
        self,
        group: bool = None,
        prior: str = None,
        block: str = None,
        include_null: bool = None,
    ) -> pd.DataFrame:
        if group is None:
            group = self.group
        if prior is None:
            prior = self.prior
        if block is None:
            block = self.block
        if include_null is None:
            include_null = self.include_null

        subset = self.optimization_data[
            (self.optimization_data["applied_policy"] == "SoftmaxPolicy")
            & (self.optimization_data["Block"].isin(block))
            & (self.optimization_data["Prior"] == prior)
            & (self.optimization_data["Group"] == group)
        ].copy(deep=True)
        if include_null:
            # random policy doesn't have prior
            subset = pd.concat(
                [
                    subset,
                    self.optimization_data[
                        (self.optimization_data["applied_policy"] == "RandomPolicy")
                        & (self.optimization_data["Block"].isin(block))
                        & (self.optimization_data["Group"] == group)
                    ].copy(deep=True),
                ]
            )

        # check that bic is same for model duplicates by cost functions
        sum_bic = (
            subset.groupby(["Model Name", "cost_function"]).sum()["bic"].reset_index()
        )
        assert np.all(sum_bic.groupby(["Model Name"]).nunique()["bic"] == 1)

        deduplicated_models = sum_bic.drop_duplicates(["Model Name"])[
            ["Model Name", "cost_function"]
        ]
        deduplicated_models_tuples = list(
            deduplicated_models.itertuples(index=False, name=None)
        )

        subset = subset[
            subset.apply(
                lambda row: (row["Model Name"], row["cost_function"])
                in deduplicated_models_tuples,
                axis=1,
            )
        ]

        return subset

    def get_trial_by_trial_likelihoods(
        self,
        group: bool = None,
        prior: str = None,
        block: str = None,
        include_null: bool = None,
    ) -> pd.DataFrame:
        if group is None:
            group = self.group
        if prior is None:
            prior = self.prior
        if block is None:
            block = self.block
        if include_null is None:
            include_null = self.include_null

        # only made for when there is one block
        assert len(block) == 1
        trial_by_trial_file = self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/data/trial_by_trial/"
            f"{self.experiment_name}_{block[0]}_{prior}.csv"
        )

        if trial_by_trial_file.is_file():
            return pd.read_csv(trial_by_trial_file, index_col=0)
        else:
            self.irl_path.joinpath(
                f"analysis/{self.experiment_subdirectory}/data/trial_by_trial/"
            ).mkdir(parents=True, exist_ok=True)
            optimization_data = self.query_optimization_data(
                group=group,
                prior=prior,
                block=block,
                include_null=include_null,
            )
            trial_by_trial_likelihoods = self.compute_trial_by_trial_likelihoods(
                optimization_data, q_path=self.q_path
            )
            trial_by_trial_likelihoods.to_csv(trial_by_trial_file)
            return trial_by_trial_likelihoods

    def compute_trial_by_trial_likelihoods(
        self,
        optimization_data: pd.DataFrame,
        q_path: Union[str, Path] = None,
    ) -> pd.DataFrame:
        if q_path is None:
            q_path = self.irl_path.joinpath("cluster/data/q_files")
        else:
            q_path = self.irl_path.joinpath(q_path)

        # load all q files
        unique_costs = {}
        for cost_function in self.cost_details.keys():
            unique_costs_rows = optimization_data[
                (optimization_data["cost_function"] == cost_function)
                & (optimization_data["applied_policy"] == "SoftmaxPolicy")
            ][self.cost_details[cost_function]["cost_parameter_args"]].drop_duplicates()
            unique_costs[cost_function] = unique_costs_rows[
                sorted(list(unique_costs_rows))
            ].to_dict("records")

        q_files = {
            get_param_string(cost_kwarg): load_q_file(
                experiment_setting=self.experiment_setting,
                cost_function=cost_function,
                cost_params=cost_kwarg,
                path=q_path,
            )
            for cost_function, cost_kwargs in unique_costs.items()
            for cost_kwarg in cost_kwargs
        }

        all_values = []

        # softmax policy

        for cost_function in self.cost_details.keys():
            subset_df = optimization_data[
                (optimization_data["applied_policy"] == "SoftmaxPolicy")
                & (optimization_data["cost_function"] == cost_function)
            ]

            for model in subset_df["Model Name"].unique():
                subset_subset_df = subset_df[subset_df["Model Name"] == model]
                unique_settings = (
                    subset_subset_df[
                        self.cost_details[cost_function]["cost_parameter_args"]
                        + ["temp"]
                    ]
                    .drop_duplicates()
                    .to_dict("records")
                )
                for unique_setting in unique_settings:
                    curr_pids = subset_subset_df[
                        subset_subset_df.apply(
                            lambda row: np.all(
                                [row[key] == val for key, val in unique_setting.items()]
                            ),
                            axis=1,
                        )
                    ]["trace_pid"]

                    subset_traces = get_trajectories_from_participant_data(
                        self.mouselab_trials[
                            self.mouselab_trials["pid"].isin(curr_pids)
                        ]
                    )
                    cost_kwargs = {
                        key: val
                        for key, val in unique_setting.items()
                        if key
                        in self.cost_details[cost_function]["cost_parameter_args"]
                    }

                    participant = SymmetricMouselabParticipant(
                        experiment_setting=self.experiment_setting,
                        num_trials=max(
                            [len(trace["actions"]) for trace in subset_traces]
                        ),
                        cost_function=eval(cost_function),
                        cost_kwargs=cost_kwargs,
                        policy_function=SoftmaxPolicy,
                        policy_kwargs={
                            "preference": q_files[get_param_string(cost_kwargs)],
                            "temp": unique_setting["temp"],
                            "noise": 0,
                        },
                    )

                    for trace in subset_traces:
                        trace["likelihood"] = [
                            np.sum(trial_vals)
                            for trial_vals in participant.compute_likelihood(trace)
                        ]
                    curr_trial_by_trial_df = traces_to_df(subset_traces)

                    curr_values = (
                        curr_trial_by_trial_df.groupby(["pid", "i_episode"])
                        .mean()["likelihood"]
                        .reset_index()
                    )
                    for key, val in unique_setting.items():
                        curr_values[[key]] = val
                    curr_values["cost_function"] = cost_function
                    curr_values["applied_policy"] = "SoftmaxPolicy"
                    curr_values["Model Name"] = model

                    all_values.append(curr_values)

        # random policy
        traces = get_trajectories_from_participant_data(self.mouselab_trials)

        participant = SymmetricMouselabParticipant(
            experiment_setting=self.experiment_setting,
            num_trials=max([len(trace["actions"]) for trace in traces]),
            policy_function=RandomPolicy,
            policy_kwargs={},
        )

        for trace in traces:
            trace["likelihood"] = [
                np.sum(trial_vals)
                for trial_vals in participant.compute_likelihood(trace)
            ]
        curr_trial_by_trial_df = traces_to_df(traces)

        curr_values = (
            curr_trial_by_trial_df.groupby(["pid", "i_episode"])
            .mean()["likelihood"]
            .reset_index()
        )
        for key, val in unique_setting.items():
            curr_values[[key]] = val
        curr_values["cost_function"] = optimization_data[
            optimization_data["applied_policy"] == "RandomPolicy"
        ]["cost_function"].unique()[0]
        curr_values["applied_policy"] = "RandomPolicy"
        for cost_param in cost_kwargs.keys():
            curr_values[cost_param] = np.nan
        curr_values["Model Name"] = optimization_data[
            optimization_data["applied_policy"] == "RandomPolicy"
        ]["Model Name"].unique()[0]

        all_values.append(curr_values)

        all_values = pd.concat(all_values)
        return all_values
