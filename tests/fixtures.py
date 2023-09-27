import json
from pathlib import Path
from mouselab.graph_utils import get_structure_properties

def load_structure_dicts(structure_file_name):
    """Load structure file."""

    structure_file = (
            Path(__file__).parents[0] / "inputs" / "structure" / f"{structure_file_name}.json"
    )
    with open(
            structure_file,
            "rb",
    ) as f:
        structure_data = json.load(f)

    structure_dicts = get_structure_properties(structure_data)
    return structure_dicts
