import colorcet as cc
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns


def set_font_sizes(SMALL_SIZE=16, MEDIUM_SIZE=20, BIGGER_SIZE=30):
    """
    Good font sizes for a poster: 24, 36, 48
    """

    plt.rcParams['text.usetex'] = True

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc("axes", titlesize=BIGGER_SIZE)


def generate_model_palette(model_names):

    static_palette = {
        model: sns.color_palette(cc.glasbey_category10, n_colors=len(model_names))[
            model_idx
        ]
        for model_idx, model in enumerate(sorted(model_names))
    }
    return static_palette


def get_static_palette(static_directory, experiment_name):
    palette_file = static_directory.joinpath(
        f"data/{experiment_name}_models_palette.pickle"
    )
    with open(palette_file, "rb") as f:
        palette = pickle.load(f)

    return palette
