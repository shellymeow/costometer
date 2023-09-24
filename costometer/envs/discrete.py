"""Provides mixins for gym environments and a new example gym environment."""
import sys
from typing import Any, Callable, Dict, Generator, List, Tuple, Union

import gym
import numpy as np
from gym import utils
from gym.envs.toy_text import CliffWalkingEnv
from gym.envs.toy_text.discrete import DiscreteEnv


class ExactSolveMixin:
    """
    This Mixin provides the functions and variables exact.py's solve function requires.

    However -- it appears since states loop back onto each other than the dynamic programming solution (Value Iteration) is more efficient than the memoization + recursion found in exact.py.
    N.B. exact.py is favored for Mouselab environments as the state space is quite large!
    """  # noqa: E501

    def __init__(self, cost: Callable = None):
        """
        Create ExactSolveMixin object.

        :param cost: additional cost function, as a function of current state, action, new state and done (e.g. boolean marking if episode/trial is over)
        """  # noqa: E501
        # need to add initial states
        self.add_initial_states()
        # and terminal state in transition and actions
        self.term_state = "__term_state__"
        self.add_terminal_state()

        if cost is None:
            self.cost = lambda old_state, next_state, done: 0
        else:
            self.cost = cost

    def actions(self, state: Union[str, int]) -> Generator[List[int], None, None]:
        """
        Add "actions" function to OpenAI gym discrete environment.

        :param state: state number or a string, in the case of the terminal state
        :return: yields available actions in state
        """
        if state == self.term_state:
            return

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            for i in range(self.action_space.n):
                yield i
        else:
            raise NotImplementedError

    def add_initial_states(self) -> None:
        """
        Add "initial_states" to OpenAI gym discrete environment.

        :return: None
        """
        if hasattr(self, "isd"):
            initial_states_and_probabilities = [
                (action_idx, prob)
                for action_idx, prob in enumerate(self.isd)
                if prob > 0
            ]
            self.initial_states = [
                action_idx for action_idx, _ in initial_states_and_probabilities
            ]
            self.initial_state_probabilities = [
                prob for _, prob in initial_states_and_probabilities
            ]
        else:
            raise ValueError("No initial_state_distrib (isd)")

    def add_terminal_state(self) -> None:
        """
        Add terminal state to OpenAI gym discrete environment.

        In unmodified OpenAI gym environments, the done variable often signifies when to stop and therefore terminal self-loops with cost might be present.

        :return: None
        """  # noqa: E501
        assert self.term_state not in self.P

        for state, state_outcomes in self.P.items():
            for action, action_outcomes in state_outcomes.items():
                # if any outcome leads to final state
                if any([action_outcome[3] for action_outcome in action_outcomes]):
                    # replace outcome with new "terminal" state
                    self.P[state][action] = [
                        (p, ns, r, done) if not done else (p, self.term_state, r, done)
                        for (p, ns, r, done) in action_outcomes
                    ]

        # add terminal state entry, not really needed
        # depending on the planning algorithm code you
        # might want to add a "loop" back to itself for
        # all actions and modify the `actions` method
        self.P[self.term_state] = {}

    def results(
        self, s: Union[str, int], a: int
    ) -> Generator[Tuple[float, int, Union[float, int]], None, None]:
        """
        Yield possible results from (state, action) pair.

        This is adapted from the parent class:

        1) removing any places where they change instance variables
        2) outputting in the format the exact.py code expects

        :param s: state to forecast results for
        :param a: action to forecast results for
        :return: all possible results in the form (probability, next_state, reward)
        """
        transitions = self.P[s][a]
        for p, s1, r, d in transitions:
            yield (p, s1, r + self.cost(s, a, s1, d))

    def step(
        self, a: int
    ) -> Tuple[Union[int, str], Union[int, float], bool, Dict[Any, Any]]:
        """
        We need to replace the base step with a version with our added costs.

        The easiest way to do this seemed to be to call results which already has costs added.
        N.B.: the mixin must come first e.g. NewGymClass(Mixin, GymEnv) in order to override the default step() method!

        :param a: current action
        :return: tuple containing (next state, reward, done, info dictionary}
        """  # noqa: E501
        prev_s = self.s
        possibilities = list(self.results(prev_s, a))
        choice_idx = np.random.choice(
            len(possibilities), p=[possibility[0] for possibility in possibilities]
        )
        p, s, r = possibilities[choice_idx]
        # update last s and last action like done in
        # https://github.com/openai/gym/blob/v0.21.0/gym/envs/toy_text/discrete.py
        self.s = s
        self.lastaction = a
        return (s, r, s == self.term_state, {"prob": p})


class RenderGridPolicyMixin:
    """Provides render_policy method for OpenAI discrete gym environments."""

    def __init__(self):
        self.policy_characters = {0: "^", 1: ">", 2: "∨", 3: "<"}

    def render_policy(self, pi: Dict[int, int]) -> None:
        """
        Render policy for grid environment.

        Assumes there are four actions and that the mapping matches:

        0. ^
        1. >
        2. ∨
        3. <

        Potential future improvement: see if we can add policy to base gym class' render (type = 'ansi')

        :param pi: policy function, as found with recursion or dynamic programming
        :return: None
        """  # noqa: E501
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            output = self.policy_characters[pi[s]]

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")


class VerySimpleGridWorld(DiscreteEnv):
    """
    This very simple grid world consists of a 3x3 grid.

    The agent starts in the top left corner and must make it to the top right corner.
    All actions are deterministic.
    """

    def __init__(self):
        nrow = 3
        ncol = 3
        self.shape = (nrow, ncol)
        nS = nrow * ncol  # noqa : N806
        nA = 4  # noqa : N806

        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        terminal_position = (0, ncol - 1)
        terminal_state = np.ravel_multi_index(terminal_position, self.shape)

        P = {}  # noqa : N806
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            for action_idx, displacement in enumerate(
                [(-1, 0), (0, 1), (1, 0), (0, -1)]
            ):
                new_position = (
                    min(max(position[0] + displacement[0], 0), nrow - 1),
                    min(max(position[1] + displacement[1], 0), ncol - 1),
                )
                new_state = np.ravel_multi_index(new_position, self.shape)
                P[s][action_idx] = [(1.0, new_state, -1, new_state == terminal_state)]

        for action in range(nA):
            P[terminal_state][action] = [(1.0, terminal_state, 0, 1)]

        super().__init__(nS, nA, P, isd)


class ModifiedVerySimpleGridWorld(
    ExactSolveMixin, VerySimpleGridWorld, RenderGridPolicyMixin
):
    """In principle, you should be able to do this with any gym environment inheriting from DiscreteEnv."""  # noqa: E501

    def __init__(
        self, cost_function: Callable = None, cost_kwargs: Dict[str, Any] = None
    ):
        VerySimpleGridWorld.__init__(self)
        # need exact solve mixin init
        # second for initial states
        if cost_kwargs is None:
            cost_kwargs = {}
        if cost_function:
            cost = cost_function(**cost_kwargs)
        else:
            cost = lambda old_state, action, curr_state, done: 0  # noqa: E731
        ExactSolveMixin.__init__(self, cost=cost)
        RenderGridPolicyMixin.__init__(self)


class ModifiedCliffWalkingEnv(ExactSolveMixin, CliffWalkingEnv, RenderGridPolicyMixin):
    def __init__(
        self, cost_function: Callable = None, cost_kwargs: Dict[str, Any] = None
    ):
        CliffWalkingEnv.__init__(self)
        # need exact solve mixin init
        # second for initial states
        if cost_kwargs is None:
            cost_kwargs = {}
        if cost_function:
            cost = cost_function(**cost_kwargs)
        else:
            cost = lambda old_state, action, curr_state, done: 0  # noqa: E731
        ExactSolveMixin.__init__(self, cost=cost)
        RenderGridPolicyMixin.__init__(self)

    def render_trajectory(self, trace: Dict[str, List]) -> None:
        """
        Render trajectory.

        Most of this method copied from:
        https://github.com/openai/gym/blob/v0.21.0/gym/envs/toy_text/cliffwalking.py

        Added red coloring to cliff and changed output for states where the trace visits.

        :param trace: trace dictionary with 'states' and 'actions' keys
        :return: None
        """  # noqa: E501
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = utils.colorize(" C ", "red", highlight=False)
            else:
                output = " o "

            if s in trace["states"]:
                action_index = max(
                    action_index
                    for action_index, state in enumerate(trace["states"])
                    if state == s
                )
                output = f" {utils.colorize(self.policy_characters[trace['actions'][action_index]], 'green', highlight=True)} "  # noqa: E501

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
