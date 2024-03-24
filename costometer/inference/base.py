"""Base inference class."""
from copy import deepcopy
from typing import Any, Dict, List

import pandas as pd

from ..utils import traces_to_df


class BaseInference:
    """Base inference class."""

    def __init__(self, traces: List[Dict[str, List]]):
        """
        Create BaseInference object.

        :param traces: the traces for which we are inferring parameters, each a dictionary with at least "actions" and "states" as fields
        """  # noqa: E501
        # save inputs
        self.traces = traces

    def run(self) -> None:
        """
        Find best parameters.

        :return:
        """
        raise NotImplementedError

    def get_best_parameters(self) -> List[Any]:
        """
        Return the best parameters for the traces.

        :return: None
        """
        raise NotImplementedError

    def get_output_df(self) -> pd.DataFrame:
        """
        Save traces as pandas dataframe, along with log likelihoods and best parameters.

        :return: dataframe of length actions * learners
        """  # noqa: E501
        traces = deepcopy(self.traces)
        trace_df = traces_to_df(traces)

        return trace_df
