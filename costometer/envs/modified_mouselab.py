"""Modifies MouselabEnv so it works like the Open AI gym discrete environments."""
import itertools
from typing import Callable, Union

from mouselab.mouselab import MouselabEnv


class ModifiedMouseLabEnv(MouselabEnv):
    """This class adds the necessary variables for applying the planning_algorithms in the costometer package."""  # noqa: E501

    @classmethod
    def new_symmetric_registered(
        cls, experiment_setting: str, seed: Union[Callable, int] = None, **kwargs
    ) -> MouselabEnv:
        """
        Add variables to make the Mouselab environment mimic the Open AI gym discrete environments.

        :param experiment_setting: experiment setting on the mouselab "registry"
        :param seed: random seed
        :param kwargs: any other MouseLabEnv arguments
        :return: MouselabEnv instance ready for planning algorithms
        """  # noqa: E501
        instance = super().new_symmetric_registered(
            experiment_setting, seed=seed, **kwargs
        )

        possibilities = [
            node.vals + tuple([node]) if hasattr(node, "sample") else tuple([node])
            for node in instance.init
        ]
        instance.P = {state: None for state in list(itertools.product(*possibilities))}
        instance.P["__term_state__"] = None
        instance.terminal_state = "__term_state__"
        return instance
