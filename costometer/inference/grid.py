"""Grid inference class."""
import itertools
from copy import deepcopy
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd
from mouselab.distributions import Categorical
from tqdm import tqdm

from ..agents.vanilla import Participant
from ..utils import adjust_state, traces_to_df
from .base import BaseInference


class GridInference(BaseInference):
    """Grid inference class."""

    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable,
        cost_parameters: Dict[str, Categorical],
        cost_function_name: str = None,
        held_constant_policy_kwargs: Dict[str, Categorical] = None,
        policy_parameters: Dict[str, Categorical] = None,
        q_files: Dict[str, Any] = None,
        verbose: bool = False,
    ):
        """
        Grid inference class.

        :param traces:
        :param participant_class:
        :param participant_kwargs:
        :param cost_function:
        :param cost_parameters:
        :param cost_function_name:
        :param held_constant_policy_kwargs:
        :param policy_parameters:
        :param q_files:
        """
        super().__init__(traces)

        self.participant_class = participant_class
        self.participant_kwargs = participant_kwargs
        self.cost_function = cost_function
        self.cost_function_name = cost_function_name
        self.cost_parameters = cost_parameters

        # policy parameters vary between agents
        if held_constant_policy_kwargs is None:
            self.held_constant_policy_kwargs = {}
        else:
            self.held_constant_policy_kwargs = held_constant_policy_kwargs
        if policy_parameters is None:
            self.policy_parameters = {}
        else:
            self.policy_parameters = policy_parameters

        if "kappa" not in self.policy_parameters:
            self.policy_parameters["kappa"] = Categorical([1], [1])
        if "gamma" not in self.policy_parameters:
            self.policy_parameters["gamma"] = Categorical([1], [1])

        self.optimization_results = None

        self.prior_probability_dict = {
            **{
                cost_parameter: dict(zip(cost_prior.vals, cost_prior.probs))
                for cost_parameter, cost_prior in self.cost_parameters.items()
            },
            **{
                policy_parameter: dict(zip(policy_prior.vals, policy_prior.probs))
                for policy_parameter, policy_prior in self.policy_parameters.items()
            },
        }
        self.optimization_space = self.get_optimization_space()

        self.verbose = verbose

    def function_to_optimize(self, config, traces, optimize=True):
        """
        Define function to optimize.

        :param config:
        :param traces:
        :param optimize:
        :return:
        """
        policy_kwargs = {
            key: config[key] for key in self.policy_parameters.keys() if key in config
        }
        cost_kwargs = {
            key: config[key] for key in self.cost_parameters.keys() if key in config
        }
        additional_params = {}

        for key, val in self.held_constant_policy_kwargs.items():
            if key == "q_function_generator":
                q_function_generator = val
                policy_kwargs["preference"] = q_function_generator(
                    cost_kwargs, policy_kwargs["kappa"], policy_kwargs["gamma"]
                )
            else:
                policy_kwargs[key] = val
                additional_params[key] = val

        participant = self.participant_class(
            **self.participant_kwargs,
            num_trials=max([len(trace["actions"]) for trace in traces]),
            cost_function=self.cost_function,
            cost_kwargs=cost_kwargs,
            policy_kwargs={
                key: val
                for key, val in policy_kwargs.items()
                if key not in ["gamma", "kappa"]
            },
        )

        result = []
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

            participant_likelihood = participant.compute_likelihood(trace)

            if optimize is True:
                # sum over actions in trial, then trials
                trial_mles = np.fromiter(map(sum, participant_likelihood), dtype=float)
                mle = np.sum(trial_mles)
                map_val = mle + np.sum(
                    [
                        np.log(prior_dict[config[param]])
                        for param, prior_dict in self.prior_probability_dict.items()
                    ]
                )

                # simulated trace, save info used to simulate data
                trace_info = {key: trace[key] for key in trace.keys() if "sim_" in key}
                result.append(
                    {
                        "loss": -mle,
                        "map": map_val,
                        "mle": mle,
                        "trace_pid": trace["pid"][0],
                        **trace_info,
                        **config,
                        **additional_params,
                    }
                )
            else:
                result.append(participant_likelihood)
        return result

    def get_optimization_space(self):
        """
        Combine inputs to obtain optimization space.

        :return:
        """
        possible_parameters = [
            [{cost_parameter: val} for val in cost_prior.vals]
            for cost_parameter, cost_prior in {
                **self.policy_parameters,
                **self.cost_parameters,
            }.items()
        ]
        config_dicts = list(itertools.product(*possible_parameters))
        # restructure from [{p1:v1},{p2:v2}... to [{p1:v1,p2:v2....
        search_space = []
        for config_setting in config_dicts:
            curr_params = {}
            for param in config_setting:
                curr_params = {**curr_params, **param}
            search_space.append(curr_params)

        return search_space

    def run(self):
        """
        Run optimization.

        :return:
        """
        self.optimization_results = []
        for config in tqdm(self.optimization_space, disable=not self.verbose):
            self.optimization_results.extend(
                self.function_to_optimize(config, traces=deepcopy(self.traces))
            )

    def get_best_parameters(self):
        """
        Return the best parameters for the traces.

        :return:
        """
        optimization_results = self.get_optimization_results()

        sim_cols = [col for col in list(optimization_results) if "sim_" in col]
        best_param_rows = optimization_results.iloc[
            optimization_results.groupby(["trace_pid"] + sim_cols).idxmin()["loss"]
        ]

        best_results = []
        for trace in self.traces:
            best_row = best_param_rows[
                np.all(
                    best_param_rows[sim_cols]
                    == [trace[sim_col] for sim_col in sim_cols],
                    axis=1,
                )
                & (best_param_rows["trace_pid"] == trace["pid"][0])
            ]
            assert len(best_row) == 1
            best_results.extend(best_row.to_dict("records"))
        return [
            {
                key: trace[key]
                for key in {**self.policy_parameters, **self.cost_parameters}
            }
            for trace in best_results
        ]

    def get_output_df(self):
        """
        Save traces as pandas dataframe, along with log likelihoods and best parameters.

        :return: dataframe of length actions * learners
        """  # noqa: E501
        traces = deepcopy(self.traces)
        for best_params, trace in zip(self.get_best_parameters(), traces):
            trace["pi"] = self.function_to_optimize(
                best_params, traces=[trace], optimize=False
            )[0]
        trace_df = traces_to_df(traces)
        return trace_df

    def get_optimization_results(self):
        """
        Get optimization results as dataframe.

        :return:
        """
        return pd.DataFrame(self.optimization_results)
