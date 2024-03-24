"""Optimization with hyperopt."""
from copy import deepcopy
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe  # noqa
from mouselab.distributions import Categorical
from scipy import stats  # noqa

from ..agents.vanilla import Participant
from ..utils import adjust_state, traces_to_df
from .base import BaseInference


class BaseOptimizerInference(BaseInference):
    """Base Optimizer optimization class."""

    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable,
        cost_parameters: Dict[str, Categorical],
        cost_function_name: str = None,
        held_constant_policy_kwargs: Dict[str, Any] = None,
        policy_parameters: Dict[str, Any] = None,
        optimization_settings: Dict[str, Any] = None,
    ):
        """
        Create BaseOptimizerInference object.

        :param traces:
        :param participant_class:
        :param participant_kwargs:
        :param cost_function:
        :param cost_parameters:
        :param cost_function_name:
        :param held_constant_policy_kwargs:
        :param policy_parameters:
        """
        super().__init__(traces)

        self.participant_class = participant_class
        self.participant_kwargs = participant_kwargs
        self.cost_function = cost_function
        self.cost_function_name = cost_function_name

        if cost_parameters is None:
            self.cost_parameters = {}
        else:
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

        if optimization_settings is None:
            self.optimization_settings = {}
        else:
            self.optimization_settings = optimization_settings

        self.optimization_results = None

    def function_to_optimize(self, config, trace, optimize=True):
        """
        Define function to optimize.

        :param config:
        :param trace:
        :param optimize:
        :return:
        """
        raise NotImplementedError

    def get_optimization_space(self):
        """
        Combine inputs to obtain optimization space.

        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        Run optimization.

        :return:
        """
        raise NotImplementedError

    def get_best_parameters(self):
        """
        Return the best parameters for the traces.

        :return:
        """
        return [result.get_best_config() for result in self.optimization_results]

    def get_output_df(self):
        """
        Save traces as pandas dataframe, along with log likelihoods and best parameters.

        :return: dataframe of length actions * learners
        """  # noqa: E501
        traces = deepcopy(self.traces)
        for best_params, trace in zip(self.get_best_parameters(), traces):
            trace["pi"] = self.function_to_optimize(best_params, trace, optimize=False)
        trace_df = traces_to_df(traces)
        return trace_df

    def get_optimization_results(self):
        """
        Get optimization results as dataframe.

        :return:
        """
        results_for_df = {}
        for optimization_result in self.optimization_results:
            results_for_df = {**results_for_df, **optimization_result.results}

        results_df = pd.DataFrame.from_dict(results_for_df).transpose()
        config_columns = results_df["config"].apply(pd.Series)
        return results_df.join(config_columns)


class HyperoptOptimizerInference(BaseOptimizerInference):
    """Optimization over grid."""

    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable = None,
        cost_parameters: Dict[str, Categorical] = None,
        held_constant_policy_kwargs: Dict[str, Categorical] = None,
        held_constant_cost_kwargs: Dict[str, Categorical] = None,
        policy_parameters: Dict[str, Categorical] = None,
        optimization_settings: Dict[str, Any] = None,
    ):
        """
        Create HyperoptOptimizerInference object.

        :param traces:
        :param participant_class:
        :param participant_kwargs:
        :param cost_function:
        :param cost_parameters:
        :param held_constant_policy_kwargs:
        :param held_constant_cost_kwargs:
        :param policy_parameters:
        :param optimization_settings:
        """
        super().__init__(
            traces=traces,
            participant_class=participant_class,
            participant_kwargs=participant_kwargs,
            cost_function=cost_function,
            cost_parameters=cost_parameters,
            held_constant_policy_kwargs=held_constant_policy_kwargs,
            policy_parameters=policy_parameters,
            optimization_settings=optimization_settings,
        )

        self.held_constant_cost_kwargs = held_constant_cost_kwargs

        self.prior_probability_dict = {
            **{
                cost_parameter: cost_prior["prior"]
                for cost_parameter, cost_prior in self.cost_parameters.items()
                if cost_parameter not in held_constant_cost_kwargs
            },
            **{
                policy_parameter: policy_prior["prior"]
                for policy_parameter, policy_prior in self.policy_parameters.items()
                if policy_parameter not in held_constant_policy_kwargs
            },
        }
        # make callable if not already a function
        self.prior_probability_dict = {
            key: eval(val) if isinstance(val, str) else val
            for key, val in self.prior_probability_dict.items()
        }

        self.optimization_space = self.get_optimization_space()
        self.best_parameters = None
        self.trials = None

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

        for key, val in self.held_constant_cost_kwargs.items():
            cost_kwargs[key] = val
            additional_params[key] = val

        for key, val in self.held_constant_policy_kwargs.items():
            if key == "q_function_generator":
                q_function_generator = val
            else:
                policy_kwargs[key] = val
                additional_params[key] = val

        policy_kwargs["preference"] = q_function_generator(
            cost_kwargs, policy_kwargs["kappa"], policy_kwargs["gamma"]
        )

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
            # trace["ground_truth"] = [
            #     adjust_ground_truth(
            #         ground_truth,
            #         policy_kwargs["gamma"],
            #         participant.mouselab_envs[0].mdp_graph.nodes.data("depth"),
            #     )
            #     for ground_truth in trace["ground_truth"]
            # ]

            # participant.agent.env
            participant_likelihood = participant.compute_likelihood(trace)

            try:
                prior_prob = np.sum(
                    [
                        prior_function(config[param])
                        for param, prior_function in self.prior_probability_dict.items()
                    ]
                )
            except KeyError:
                # sometimes hyperopt seems to suggest values not in hp.choice
                return {"loss": float("inf"), "status": STATUS_FAIL}

            if optimize is True:
                # sum over actions in trial, then trials
                trial_mles = np.fromiter(map(sum, participant_likelihood), dtype=float)
                mle = np.sum(trial_mles)
                map_val = mle + prior_prob

                # simulated trace, save info used to simulate data
                trace_info = {key: trace[key] for key in trace.keys() if "sim_" in key}
                result.append(
                    {
                        "map_val": map_val,
                        "mle": mle,
                        "trace_pid": trace["pid"][0],
                        **trace_info,
                        **config,
                        **additional_params,
                    }
                )
            else:
                result.append(participant_likelihood)

        if optimize is False:
            return result
        else:
            return {
                "loss": -np.sum([res["map_val"] for res in result]),
                "result": result,
                "status": STATUS_OK,
            }

    def get_optimization_space(self):
        """
        Combine inputs to obtain optimization space.

        :return:
        """
        search_space = {
            **{
                cost_parameter: eval(cost_prior["search_space"])
                for cost_parameter, cost_prior in self.cost_parameters.items()
                if cost_parameter not in self.held_constant_cost_kwargs
            },
            **{
                policy_parameter: eval(policy_prior["search_space"])
                for policy_parameter, policy_prior in self.policy_parameters.items()
                if policy_parameter not in self.held_constant_policy_kwargs
            },
        }
        return search_space

    def run(self):
        """
        Run optimization.

        :return:
        """
        trials = Trials()
        fmin(
            lambda config: self.function_to_optimize(
                config, traces=deepcopy(self.traces)
            ),
            space=self.optimization_space,
            algo=tpe.suggest,
            trials=trials,
            **self.optimization_settings,
        )
        self.best_parameters = trials.best_trial
        self.trials = trials
        return self.best_parameters

    def get_best_parameters(self):
        """
        Return the best parameters for the traces.

        :return:
        """
        return self.best

    def get_output_df(self):
        """
        Save traces as pandas dataframe, along with log likelihoods and best parameters.

        :return: dataframe of length actions * learners
        """  # noqa: E501
        return pd.DataFrame.from_dict(
            [trial["result"]["result"][0] for trial in self.trials.trials]
        )

    def get_optimization_results(self):
        """
        Get optimization results as dataframe.

        :return:
        """
        if not self.trials:
            # First optimize with the run method
            return None
        else:
            return {
                "trials": self.trials,
                "res": self.function_to_optimize(
                    {
                        param: param_val[0]
                        for param, param_val in self.best_parameters["misc"][
                            "vals"
                        ].items()
                    },
                    deepcopy(self.traces),
                    optimize=False,
                ),
            }
