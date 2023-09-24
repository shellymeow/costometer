from setuptools import setup

setup(
    name="costometer",
    version="0.0.1",
    packages=[
        "costometer",
        "costometer.agents",
        "costometer.envs",
        "costometer.inference",
        "costometer.planning_algorithms",
        "costometer.utils",
    ],
    url="",
    license="",
    author="Valkyrie Felso",
    author_email="",
    description="Code to apply Bayesian inverse reinforcement "
    "learning to Mouselab MDP and OpenAI gym environments",
    setup_requires=["wheel"],
    install_requires=[
        "colorcet",
        "dill",
        "gym==0.21.0",  # unfortunately DiscreteEnv class was removed in 0.22.0: https://github.com/openai/gym/pull/2514 # noqa
        "hyperopt",
        "mouselab @ git+https://github.com/RationalityEnhancementGroup/mouselab-mdp-tools.git@main#egg=mouselab",  # noqa
        "numpy",
        "pandas",
        "statsmodels",
        "tqdm",
        "blosc",
    ],
)
