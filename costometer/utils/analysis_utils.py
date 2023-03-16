"""Utility functions for MAP calculation, priors and finding the best parameters."""
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from more_itertools import powerset
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import SoftmaxPolicy
from scipy import stats  # noqa
from scipy.stats import rv_continuous
from statsmodels.tools.eval_measures import bic

from costometer.agents import SymmetricMouselabParticipant
from costometer.utils.cost_utils import adjust_state, get_state_action_values
from costometer.utils.plotting_utils import generate_model_palette
from costometer.utils.trace_utils import get_trajectories_from_participant_data


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
        # uniform should always be provided
        for subset in powerset(priors["uniform"]):
            # subset dataframe
            curr_data = (
                df[
                    df.apply(
                        lambda row: sum(
                            row[cost_param]
                            == cost_details["constant_values"][cost_param]
                            for cost_param in list(subset)
                        )
                        == len(list(subset)),
                        axis=1,
                    )
                ]
                .copy(deep=True)
                .reset_index(drop=True)
            )

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

            best_param_rows = curr_data.loc[  #
                curr_data.groupby(["trace_pid"] + sim_cols).idxmax(numeric_only=True)[
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

        non_temp_params = set(
            list(cost_details["constant_values"]) + additional_params
        ) - set(["temp"])
        for cost_parameter_arg in non_temp_params:
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

        dfs = {}
        for session in self.sessions:
            matching_files = self.irl_path.glob(f"data/processed/{session}/*.csv")
            for matching_file in matching_files:
                curr_df = pd.read_csv(matching_file, index_col=0)
                curr_df["session"] = session

                if matching_file.stem not in dfs:
                    dfs[matching_file.stem] = [curr_df]
                else:
                    dfs[matching_file.stem].append(curr_df)

        self.dfs = {file_type: pd.concat(df_list) for file_type, df_list in dfs.items()}

        if not self.simulated:
            self.load_session_details()
        else:
            # create 'num_clicks'
            self.dfs["mouselab-mdp"]["num_clicks"] = 1

            # simulated data's block is always test
            self.dfs["mouselab-mdp"]["block"] = "test"

            # only keep relevant columns
            simulated_cols = [
                col for col in list(self.dfs["mouselab-mdp"]) if "sim_" in col
            ]
            self.dfs["mouselab-mdp"] = self.dfs["mouselab-mdp"][
                simulated_cols + ["pid", "block", "num_clicks", "session"]
            ].copy(deep=True)

            self.session_details = {
                session: {
                    "experiment_setting": session.split("/")[1],
                }
                for session in self.sessions
            }

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

        yaml_path = self.irl_path.joinpath(
            f"data/inputs/yamls/experiment_settings/{self.experiment_setting}.yaml"
        )
        with open(yaml_path, "r") as stream:
            self.experiment_details = yaml.safe_load(stream)

        self.optimization_data = self.load_optimization_data()

        if not self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/data/"
            f"{self.experiment_name}_models_palette.pickle"
        ).is_file():
            static_palette = generate_model_palette(self.model_name_mapping.values())
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
        yaml_file = self.irl_path.joinpath(
            f"data/inputs/yamls/cost_functions/{self.cost_function}.yaml"
        )
        with open(str(yaml_file), "r") as stream:
            self.cost_details = yaml.safe_load(stream)

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
        self.model_name_mapping = {}
        mle_and_map_files = [
            (
                session,
                self.irl_path.joinpath(
                    f"data/processed/{session}/{self.cost_function}"
                    f"/mle_and_map"
                    f"{'_' + self.block if self.block != 'test' else ''}_{pid}.pickle"
                ),
            )
            for session, pid in self.dfs["mouselab-mdp"][["session", "pid"]]
            .drop_duplicates()
            .values
        ]
        for session, mle_and_map_file in mle_and_map_files:
            with open(
                mle_and_map_file,
                "rb",
            ) as f:
                data = pickle.load(f)
            full_dfs.extend(
                [
                    {
                        **random_record,
                        f"map_{prior}": random_record["mle"],
                        "prior": prior,
                        "model": "None",
                        "Model Name": "Null",
                        "session": session,
                        "Number Parameters": 0,
                    }
                    for random_record in data["RandomPolicy"].to_dict("records")
                    for prior in data["SoftmaxPolicy"].keys()
                ]
            )
            for prior, prior_dict in data["SoftmaxPolicy"].items():
                all_params = max(prior_dict, key=len)
                for model, model_df in prior_dict.items():
                    must_contain = set(all_params) - set(
                        self.cost_details["constant_values"]
                    )
                    # in some cases, if we used a larger base cost model we will have
                    # an entry with param X held constant and not
                    # (when it always was for this cost function)
                    if must_contain.issubset(set(model)):
                        # model is held constant parameters
                        varied_parameters = set(all_params) - set(model)
                        number_parameters = len(varied_parameters)
                        cost_params_in_model = varied_parameters.intersection(
                            set(self.cost_details["cost_parameter_args"])
                        )
                        additional_params_in_model = varied_parameters.difference(
                            set(self.cost_details["cost_parameter_args"])
                        )

                        if len(cost_params_in_model) > 0:
                            model_name = (
                                "$"
                                + ", ".join(
                                    [
                                        self.cost_details["latex_mapping"][param]
                                        for param in sorted(cost_params_in_model)
                                    ]
                                )
                                + "$"
                            )
                        else:
                            model_name = "Null (Given Costs)"

                        if len(additional_params_in_model) > 0:
                            model_name = (
                                model_name
                                + " with $"
                                + ", ".join(
                                    [
                                        self.cost_details["latex_mapping"][param]
                                        for param in sorted(additional_params_in_model)
                                    ]
                                )
                                + "$"
                            )

                        self.model_name_mapping[
                            tuple(param for param in sorted(model))
                        ] = model_name
                        full_dfs.extend(
                            [
                                {
                                    **softmax_record,
                                    "prior": prior,
                                    "model": model,
                                    "Model Name": model_name,
                                    "session": session,
                                    "Number Parameters": number_parameters,
                                }
                                for softmax_record in model_df.to_dict("records")
                            ]
                        )

        full_df = pd.DataFrame(full_dfs)
        # delete old index column, if needed
        if "index" in full_df:
            del full_df["index"]
        # map may not have same prior, delete the one from the grid search
        if "map" in full_df:
            del full_df["map"]

        mouselab_data = self.dfs["mouselab-mdp"]
        # human data does not include terminal actions in num clicks
        if not self.simulated:
            mouselab_data["num_clicks"] = (
                mouselab_data["num_clicks"] + 1
            )  # add terminal action
        full_df = self.join_optimization_df_and_processed(
            optimization_df=full_df,
            processed_df=mouselab_data[
                mouselab_data["block"].isin(self.block.split(","))
            ]
            .groupby(["pid"], as_index=False)
            .sum(),
            variables_of_interest=["num_clicks"],
        )

        full_df["bic"] = full_df.apply(
            lambda row: bic(
                llf=row[f"map_{self.prior}"],
                nobs=row["num_clicks"],
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

    @staticmethod
    def join_optimization_df_and_processed(
        optimization_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        variables_of_interest: List[str] = None,
    ) -> pd.DataFrame:
        if all(var in processed_df for var in variables_of_interest):
            merged_df = optimization_df.merge(
                processed_df[["pid", *variables_of_interest]],
                left_on=[
                    "trace_pid",
                ],
                right_on=["pid"],
                how="left",
            )
            # delete pid in case we want to merge additional dataframes
            # in the future
            del merged_df["pid"]
            return merged_df
        elif all(var in optimization_df for var in variables_of_interest):
            merged_df = processed_df.merge(
                optimization_df[["trace_pid", *variables_of_interest]],
                left_on=[
                    "pid",
                ],
                right_on=["trace_pid"],
                how="left",
            )
            # delete pid in case we want to merge additional dataframes
            # in the future
            del merged_df["trace_pid"]
            return merged_df

    def get_trial_by_trial_likelihoods(
        self,
    ) -> pd.DataFrame:
        trial_by_trial_file = self.irl_path.joinpath(
            f"analysis/methods/static/data/trial_by_trial/"
            f"{self.experiment_name}.pkl"
        )

        if trial_by_trial_file.is_file():
            with open(trial_by_trial_file, "rb") as f:
                all_trial_by_trial = pickle.load(f)
        else:
            self.irl_path.joinpath(
                "analysis/methods/static/data/trial_by_trial/"
            ).mkdir(parents=True, exist_ok=True)

            all_trial_by_trial = {}
            for excluded_parameters in self.trial_by_trial_models:
                curr_trial_by_trial = self.compute_trial_by_trial_likelihoods(
                    excluded_parameters=excluded_parameters,
                )
                all_trial_by_trial[excluded_parameters] = curr_trial_by_trial

            with open(trial_by_trial_file, "wb") as f:
                pickle.dump(all_trial_by_trial, f)

        avg_trial = []
        for excluded_parameters in self.trial_by_trial_models:
            participant_lik_trial_dicts = all_trial_by_trial[excluded_parameters]
            if excluded_parameters == "":
                model_name = self.model_name_mapping[()]
            else:
                model_name = self.model_name_mapping[
                    tuple(excluded_parameters.split(","))
                ]

            avg_trial.extend(
                [
                    [
                        pid,
                        sum([np.exp(action_ll) for action_ll in trial_ll])
                        / len(trial_ll),
                        model_name,
                        excluded_parameters == self.excluded_parameters,
                        trial_num,
                    ]
                    for pid, all_ll in participant_lik_trial_dicts.items()
                    for trial_num, trial_ll in enumerate(all_ll)
                ]
            )

        trial_by_trial_df = pd.DataFrame(
            avg_trial, columns=["pid", "avg", "Model Name", "best_model", "i_episode"]
        )

        return trial_by_trial_df

    def compute_trial_by_trial_likelihoods(
        self, excluded_parameters: str = None
    ) -> Dict[int, List[Any]]:
        if excluded_parameters is None:
            excluded_parameters = self.excluded_parameters

        optimization_data = self.query_optimization_data()
        optimization_data = optimization_data[
            optimization_data["applied_policy"] == "SoftmaxPolicy"
        ]
        if excluded_parameters == "":
            optimization_data = optimization_data[
                optimization_data["model"].apply(lambda model: set(model) == set())
            ].copy(deep=True)
        else:
            optimization_data = optimization_data[
                optimization_data["model"].apply(
                    lambda model: set(model) == set(excluded_parameters.split(","))
                )
            ].copy(deep=True)

        experiment_setting = self.experiment_setting

        with open(
            self.irl_path.joinpath(
                f"data/inputs/exp_inputs/structure/"
                f"{self.experiment_details['structure']}.json"
            ),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)

        q_function_generator = (
            lambda cost_parameters, a, g: get_state_action_values(  # noqa : E731
                experiment_setting=experiment_setting,
                bmps_file="Myopic_VOC",
                bmps_path=self.irl_path.joinpath("cluster/parameters/bmps"),
                cost_function=eval(self.cost_details["cost_function"]),
                cost_parameters=cost_parameters,
                structure=structure_dicts,
                env_params=self.cost_details["env_params"],
                kappa=a,
                gamma=g,
            )
        )

        pid_to_best_params = (
            optimization_data[
                list(self.cost_details["constant_values"]) + ["trace_pid"]
            ]
            .set_index("trace_pid")
            .to_dict("index")
        )

        trial_by_trial = {}
        for pid, config in pid_to_best_params.items():
            traces = get_trajectories_from_participant_data(
                self.dfs["mouselab-mdp"][self.dfs["mouselab-mdp"]["pid"] == pid],
                experiment_setting=experiment_setting,
                include_last_action=self.cost_details["env_params"][
                    "include_last_action"
                ],
            )

            policy_kwargs = {
                key: val
                for key, val in config.items()
                if key not in self.cost_details["cost_parameter_args"]
            }

            cost_kwargs = {
                key: val
                for key, val in config.items()
                if key in self.cost_details["cost_parameter_args"]
            }

            policy_kwargs["noise"] = 0
            policy_kwargs["preference"] = q_function_generator(
                cost_kwargs, policy_kwargs["kappa"], policy_kwargs["gamma"]
            )

            participant = SymmetricMouselabParticipant(
                experiment_setting=experiment_setting,
                policy_function=SoftmaxPolicy,
                additional_mouselab_kwargs={
                    "mdp_graph_properties": structure_dicts,
                    **self.cost_details["env_params"],
                },
                num_trials=max([len(trace["actions"]) for trace in traces]),
                cost_function=eval(self.cost_details["cost_function"]),
                cost_kwargs=cost_kwargs,
                policy_kwargs={
                    key: val
                    for key, val in policy_kwargs.items()
                    if key not in ["gamma", "kappa"]
                },
            )

            for trace in traces:
                trace["states"] = [
                    [
                        adjust_state(
                            state,
                            policy_kwargs["gamma"],
                            participant.mouselab_envs[0].mdp_graph.nodes.data("depth"),
                            True,
                        )
                        for state in trial
                    ]
                    for trial in trace["states"]
                ]

                trial_by_trial[pid] = participant.compute_likelihood(trace)

                sum_trial_by_trial = sum(
                    [
                        sum(trial_ll)
                        for block, trial_ll in zip(trace["block"], trial_by_trial[pid])
                        if block in self.block.split(",")
                    ]
                )
                assert (
                    sum_trial_by_trial
                    - optimization_data[optimization_data["trace_pid"] == pid][
                        "mle"
                    ].values[0]
                ) < 1e-3

        return trial_by_trial

    def load_hdi_ranges(self, excluded_parameters: str = None):
        if excluded_parameters is None:
            excluded_parameters = self.excluded_parameters

        if excluded_parameters != "":
            file_end = "_" + excluded_parameters
        else:
            file_end = ""

        hdi_ranges = {}
        for session, pid in (
            self.dfs["mouselab-mdp"][["session", "pid"]].drop_duplicates().values
        ):
            hdi_file = self.irl_path.joinpath(
                f"cluster/data/marginal_hdi/{self.cost_function}/{session}/"
                f"{self.block}_{self.prior}_hdi_{pid}{file_end}.pickle"
            )
            with open(
                hdi_file,
                "rb",
            ) as f:
                hdi_ranges[pid] = pickle.load(f)

        return hdi_ranges

    def query_optimization_data(
        self,
        prior: str = None,
        include_null: bool = None,
        excluded_parameters: str = None,
    ) -> pd.DataFrame:
        if prior is None:
            prior = self.prior
        if include_null is None:
            include_null = self.include_null

        subset = self.optimization_data[
            (self.optimization_data["applied_policy"] == "SoftmaxPolicy")
            & (self.optimization_data["prior"] == prior)
        ].copy(deep=True)
        if include_null:
            # random policy doesn't have prior
            subset = pd.concat(
                [
                    subset,
                    self.optimization_data[
                        (self.optimization_data["applied_policy"] == "RandomPolicy")
                        & (self.optimization_data["prior"] == prior)
                    ].copy(deep=True),
                ]
            )

        if excluded_parameters is None:
            return subset
        elif excluded_parameters == "":
            return subset[
                subset["model"].apply(lambda model: set(model) == set())
            ].copy(deep=True)
        else:
            return subset[
                subset["model"].apply(
                    lambda model: set(model) == set(excluded_parameters.split(","))
                )
            ].copy(deep=True)
