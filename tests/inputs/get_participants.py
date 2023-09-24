import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mouselab.graph_utils import get_structure_properties
from mouselab.mouselab import MouselabEnv

from costometer.utils import get_trajectories_from_participant_data

if __name__ == "__main__":
    all_dfs = []
    for fake_participant_idx, fake_participant_file in enumerate(
        Path(__file__).parents[0].glob("fake_data/*.json")
    ):
        with open(fake_participant_file, "r") as f:
            fake_participant_data = json.load(f)
            data = pd.DataFrame(fake_participant_data)
            data["pid"] = fake_participant_idx
            all_dfs.append(data)

    all_data = pd.concat(all_dfs)
    all_data = all_data.rename(
        columns={"actionTimes": "action_times", "stateRewards": "state_rewards"}
    )
    all_data[all_data["trial_type"] == "mouselab-mdp"].to_csv(
        Path(__file__).parents[0].joinpath("fake_data/mouselab_mdp.csv")
    )

    trajectories = get_trajectories_from_participant_data(
        all_data[all_data["trial_type"] == "mouselab-mdp"],
        experiment_setting="high_increasing",
        include_last_action=False,
    )

    structure = {
        "layout": {
            "0": [0, 0],
            "1": [0, -1],
            "2": [0, -2],
            "3": [1, -2],
            "4": [-1, -2],
            "5": [1, 0],
            "6": [2, 0],
            "7": [2, -1],
            "8": [2, 1],
            "9": [-1, 0],
            "10": [-2, 0],
            "11": [-2, -1],
            "12": [-2, 1],
        },
        "initial": "0",
        "graph": {
            "0": {"up": [0, "1"], "right": [0, "5"], "left": [0, "9"]},
            "1": {"up": [0, "2"]},
            "2": {"right": [0, "3"], "left": [0, "4"]},
            "3": {},
            "4": {},
            "5": {"right": [0, "6"]},
            "6": {"up": [0, "7"], "down": [0, "8"]},
            "7": {},
            "8": {},
            "9": {"left": [0, "10"]},
            "10": {"up": [0, "11"], "down": [0, "12"]},
            "11": {},
            "12": {},
        },
    }

    mdp_graph_properties = get_structure_properties(structure)

    Path(__file__).parents[1].joinpath("outputs/bias_plots").mkdir(
        parents=True, exist_ok=True
    )

    for participant_idx, participant in enumerate(trajectories):
        for trial_idx, ground_truth in enumerate(participant["ground_truth"]):
            actions = participant["actions"][trial_idx]
            env = MouselabEnv.new_symmetric_registered(
                "high_increasing",
                ground_truth=ground_truth,
                mdp_graph_properties=mdp_graph_properties,
            )
            for action in actions[:-1]:
                env.step(action)
            plt.plot()
            env._render(use_networkx=True)
            plt.title(f"PID: {participant_idx}, trial: {trial_idx}")
            plt.savefig(
                Path(__file__)
                .parents[1]
                .joinpath(
                    f"outputs/bias_plots/pid_{participant_idx}_trial_{trial_idx}.png"
                )
            )
