"""Provides participant classes for use with gym environments."""
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import gym
import numpy as np
from mouselab.agents import Agent
from mouselab.distributions import Categorical
from mouselab.mouselab import MouselabEnv

from costometer.utils.trace_utils import adjust_ground_truth, adjust_state


class Participant:
    def __init__(
        self,
        envs: List[gym.Env] = None,
        num_trials: int = None,
        ground_truths: List[Any] = None,
        trial_ids: List[Any] = None,
        cost_function: Callable = None,
        cost_kwargs: Dict[str, Any] = None,
        trace: Dict[str, List] = None,
        policy_function: Callable = None,
        policy_kwargs: Dict[str, Any] = None,
        kappa: int = 1,
        gamma: int = 1,
    ):
        """
        Create generic gym participant.

        :param envs: list of gym environments
        :param num_trials: Number of trials for the participant
        :param ground_truths:  Ground truth rewards (useful to simulate exact trials participants see)
        :param trial_ids: Trial IDs when ground truth provided
        :param cost_function: cost function, as function
        :param cost_kwargs: keyword arguments for cost function, (e.g., {'distance_weight' : 10})
        :param trace: trajectory for participant (real or simulated)
        :param policy_function: assumed policy of participant (e.g., softmax, optimal, random)
        :param policy_kwargs: keyword arguments for policy function (e.g., {'temperature' : 2})
        :param kappa
        :param gamma
        """  # noqa: E501
        # save and check that num trials and ground truths agree
        self._check_for_pipeline_info(num_trials, ground_truths, trial_ids)

        self.envs = envs

        self.agent = Agent()
        self.agent.register(self.envs)

        self.policy_function = policy_function
        # policy kwargs, e.g. beta for softmax policy
        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_kwargs = policy_kwargs

        # set cost_function and cost_kwargs
        self.cost_function = cost_function
        if cost_kwargs is None:
            cost_kwargs = {}
        self.cost_kwargs = cost_kwargs

        if policy_function is not None:
            pol = self.policy_function(**self.policy_kwargs)
            self.agent.register(pol)

        # initialize experiment data as {} if not attached
        # (note mutable objects SHOULD NOT be default values which is why we do this)
        self.trace = trace

        self.kappa = kappa
        self.gamma = gamma
        self.agent.adjusted = True if kappa == 1 and gamma == 1 else False

    def simulate_trajectory(
        self, render: bool = False, pbar: bool = False, force: bool = False
    ):
        """
        Simulate trajectory from q_function.

        :param render: whether to render the environment (untested)
        :param pbar: progress bar while simulating trajectories
        :param force: whether to force save
        :return: Nothing, saves actions and rewards to object
        """
        if not self.agent.adjusted:
            for env in self.agent.env:
                env._state = adjust_state(
                    env._state,
                    self.gamma,
                    env.mdp_graph.nodes.data("depth"),
                    len(env._state) > env.term_action,
                )
                env.ground_truth = adjust_ground_truth(
                    env.ground_truth, self.gamma, env.mdp_graph.nodes.data("depth")
                )
                env.power_utility = self.kappa
            self.agent.adjusted = True

        trace = self.agent.run_many(pbar=pbar, render=render)

        trace["ground_truth"] = self.ground_truths
        trace["trial_id"] = self.trial_ids

        if (self.trace is not None) and not force:
            raise ValueError(
                "Trace already attached, if you need to attach this set force = True"
            )
        else:
            self.trace = trace

        return trace

    def compute_likelihood(self, trace: Dict[str, List]) -> List[List[float]]:
        """
        Get (log) likelihood of trace.

        :param trace: trajectory trace as dictionary, must at least include states and actions
        :return: list of lists containing log likelihoods for an action in a trial
        """  # noqa: E501
        logliks = [[] for _ in trace["states"]]

        # sort of a hack to re-start count if needed
        self.agent.i_episode = 0

        # a trial here is often called an episode elsewhere
        for trial_idx, states in enumerate(trace["states"]):
            # get actions for this trial
            actions = trace["actions"][trial_idx]

            # loop through actions, getting likelihoods
            for action_idx, state in enumerate(states):
                # if  last state is terminal state, don't continue
                if state == "__term_state__":
                    pass
                else:
                    action = actions[action_idx]
                    # get action probabilities according to policy
                    action_distribution = self.agent.policy.action_distribution(state)
                    # append likelihood of current action to trial likelihoods
                    logliks[trial_idx].append(np.log(action_distribution[action]))
            self.agent.i_episode += 1

        return logliks

    def __deepcopy__(self, memo: Dict[Any, Any]):
        """Deepcopy, from https://stackoverflow.com/a/15774013."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _check_for_pipeline_info(
        self,
        num_trials: int,
        ground_truths: List[Any],
        trial_ids: List[Any],
    ):
        # first check either variable is set
        if (ground_truths is None) and (num_trials is None):
            raise ValueError("Either ground_truths or num_trials must be set.")

        # set num_trials
        if num_trials is None:
            self.num_trials = len(ground_truths)
        else:
            self.num_trials = num_trials

        # set ground truths
        if ground_truths is None:
            self.ground_truths = [None] * self.num_trials
        else:
            self.ground_truths = ground_truths

        # set trial ids
        if trial_ids is None:
            self.trial_ids = [None] * self.num_trials
        else:
            self.trial_ids = trial_ids

        if (
            len(
                np.unique(
                    [self.num_trials, len(self.trial_ids), len(self.ground_truths)]
                )
            )
            > 1
        ):
            raise ValueError(
                "Provided number of trials, trial ids,"
                " and ground truths must be consistent."
            )


class SymmetricMouselabParticipant(Participant):
    def __init__(
        self,
        experiment_setting: str,
        mouselab_envs: List[MouselabEnv] = None,
        additional_mouselab_kwargs: Dict[str, Any] = None,
        num_trials: int = None,
        ground_truths: List[List[Union[int, Categorical]]] = None,
        trial_ids: List[Union[int, str]] = None,
        cost_function: Callable = None,
        cost_kwargs: Dict[str, Any] = None,
        trace: Dict[str, List] = None,
        policy_function: Callable = None,
        policy_kwargs: Dict[str, Any] = None,
        kappa: int = 1,
        gamma: int = 1,
    ):
        """
        Participant class specifically for symmetric versions of the Mouselab MDP environment.

        :param experiment_setting: registered experiment setting in mouselab.envs.registry registry
        :param mouselab_envs: Mouselab MDP environments, if already constructed
        :param additional_mouselab_kwargs: additional keyword arguments for constructing Mouselab MDP environments (e.g. different final reward calculation, structure)
        :param num_trials: Number of trials for the participant
        :param ground_truths:  Ground truth rewards (useful to simulate exact trials participants see)
        :param trial_ids: Trial IDs when ground truth provided
        :param cost_function: See parent class
        :param cost_kwargs: See parent class
        :param trace: See parent class
        :param policy_function: See parent class
        :param policy_kwargs: See parent class
        :param kappa
        :param gamma
        """  # noqa: E501
        if cost_kwargs is None:
            cost_kwargs
        # if cost function not None, combine with kwargs for envs
        if cost_function:
            mouselab_cost_function = cost_function(**cost_kwargs)
        # else set to -1 just for mouselab
        else:
            mouselab_cost_function = -1

        if additional_mouselab_kwargs is None:
            additional_mouselab_kwargs = {}

        # generate pipeline of mouselab envs for symmetric environment
        if mouselab_envs is None:
            self.mouselab_envs = [
                MouselabEnv.new_symmetric_registered(
                    experiment_setting,
                    ground_truth=ground_truths[trial_idx] if ground_truths else None,
                    cost=mouselab_cost_function,
                    **additional_mouselab_kwargs
                )
                for trial_idx in range(num_trials if num_trials else len(ground_truths))
            ]
        # otherwise if num_trials exists and does not match number of envs
        elif (num_trials) and (len(mouselab_envs) != num_trials):
            raise ValueError(
                "Number of trials does not correspond to "
                "length of provided mouselab environments"
            )
        elif (ground_truths) and (len(mouselab_envs) != len(ground_truths)):
            raise ValueError(
                "Length of provided ground truths does not correspond "
                "to length of provided mouselab environments"
            )
        else:
            self.mouselab_envs = mouselab_envs

        if policy_kwargs is None:
            policy_kwargs = {}

        super().__init__(
            envs=self.mouselab_envs,
            num_trials=num_trials,
            ground_truths=ground_truths,
            trial_ids=trial_ids,
            cost_function=cost_function,
            cost_kwargs=cost_kwargs,
            trace=trace,
            policy_function=policy_function,
            policy_kwargs=policy_kwargs,
            kappa=kappa,
            gamma=gamma,
        )
