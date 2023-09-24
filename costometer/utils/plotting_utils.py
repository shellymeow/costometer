import shutil

import colorcet as cc
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns


def set_font_sizes(small_size=16, medium_size=20, bigger_size=30):
    """
    Set font sizes.

    NOTE: Good font sizes for a poster: 24, 36, 48.
    """
    # only use tex if on system
    plt.rcParams["text.usetex"] = not shutil.which("tex")

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
    plt.rc("axes", titlesize=bigger_size)


def generate_model_palette(model_names):
    static_palette = {
        model: sns.color_palette(cc.glasbey_category10, n_colors=len(model_names))[
            model_idx
        ]
        for model_idx, model in enumerate(sorted(model_names, reverse=True))
    }
    return static_palette


def get_static_palette(static_directory, experiment_name):
    palette_file = (
        static_directory / "data" / f"{experiment_name}_models_palette.pickle"
    )
    with open(palette_file, "rb") as f:
        palette = pickle.load(f)

    return palette
