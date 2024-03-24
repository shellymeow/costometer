from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class AnalysisDetails(BaseModel):
    """Analysis Details"""

    # Which experimental sessions to include
    # (real or simulated)
    sessions: List[str]

    # Whether this experiment is simulated
    simulated: bool

    # For analyses related to structure understanding,
    # names of the post-test quiz questions
    post_quizzes: Optional[List[str]] = None

    # Cost function name
    cost_function: str

    # Parameter prior name
    prior: str

    # Parameters to be excluded
    excluded_parameter_str: str

    @property
    def excluded_parameters(self) -> Tuple[str]:
        if self.excluded_parameter_str == "":
            return tuple()
        return tuple(sorted(self.excluded_parameter_str.split(",")))

    # Parameters to definitely include
    # TODO: do we actually use this
    included_parameters: List[str]

    # Experimental blocks to include
    # (e.g., depending on analysis we may not include
    # first training block)
    block: str

    @property
    def blocks(self) -> List[str]:
        return self.block.split(",")

    # Extra text to append to plot titles
    title_extras: Optional[str] = None

    # Which run to use for PPC
    # Should be chosen to be best model
    simulated_param_run: Optional[str] = None

    # Trial-by-trial models which we will compare
    # (in my case, manually selected based on model selection)
    # e.g., "gamma,kappa", "kappa"
    trial_by_trial_models: Optional[List[str]] = None

    # Mapping so we can use above, e.g.,
    # {"gamma,kappa" : ["gamma" , "kappa"], "kappa" : ["kappa"]}
    @property
    def trial_by_trial_models_mapping(self) -> Dict[str, List[str]]:
        return {
            excluded_parameters_string: tuple(
                sorted(excluded_parameters_string.split(","))
            )
            if excluded_parameters_string != ""
            else tuple()
            for excluded_parameters_string in self.trial_by_trial_models
        }

    palette_name: Optional[str] = None

    loadings: Optional[str] = None


class CostEnvironmentDetails(BaseModel):
    """Data model for Mouselab Environment details."""

    # whether to include last action in model
    # (may be needed to satisfy Markov property)
    include_last_action: bool

    # what data to use from last action
    # TODO: better documentation on what is possible here
    last_action_info: str = "distance"


class CostDetails(BaseModel):
    """Data model for Cost Function / Experiment details."""

    # cost function name
    cost_function_name: str
    # actual cost function
    cost_function: str
    # arguments to cost function: could be found with inspect.signature()
    cost_parameter_args: list
    # Prettier names for cost parameters, ordered same as in function signature
    cost_parameter_names: list
    # mapping of cost parameter arg -> latex variable
    latex_mapping: dict
    # what value is considered "constant", for considering nested models
    # e.g., for a parameter like gamma this will be 1,
    # but for most added costs it will be 0
    constant_values: dict
    # relevant environmental parameters
    # that may be needed for cost function
    env_params: CostEnvironmentDetails


class ExperimentDetails(BaseModel):
    ground_truth_file: str
    node_classification: Dict[str, List[int]]
    structure: str


class SimulatedSessionDetails(BaseModel):
    experiment_setting: str


class SessionDetails(BaseModel):
    database_key: Optional[str] = None
    participants_to_remove: Optional[List] = None
    bonus_function: Optional[str] = None
    simulated: bool
    sessions: List[str]
    experiment_setting: str
    COST: Optional[Dict[str, List[int]]] = None
    DEPTH: Optional[Dict[str, List[int]]] = None
    html_survey_names: Optional[Dict[Any, Any]] = None
    num_parts: Optional[Any] = None
    old_experiment: bool
    manual_age_mapping: Optional[Dict] = None
    experiment_specific_mapping: Optional[Dict] = None
    trials_per_block: Dict[str, int]
    ranges_to_extract: Any
    mouselab_column_identifier: Optional[str] = None
    mouselab_mapping: Optional[Dict[str, str]] = None
    max_attempts: Optional[Dict[str, int]] = None
    passing_score: Optional[Dict[str, int]] = None
    mouselab_quiz_solutions: Optional[Dict[str, Dict[str, Any]]] = None
