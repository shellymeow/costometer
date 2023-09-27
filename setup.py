from setuptools import setup

setup(
    name="costometer",
    version="0.0.2",
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
        "blosc",
        "colorcet",
        "dill",
        # "gym==0.21.0",  # unfortunately DiscreteEnv class was removed in 0.22.0: https://github.com/openai/gym/pull/2514 # noqa
        # plus commit that fixes typo: https://github.com/openai/gym/issues/3202
        "gym @ git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40",
        "hyperopt",
        "mouselab @ git+https://github.com/RationalityEnhancementGroup/mouselab-mdp-tools.git@dev#egg=mouselab",  # noqa
        "numpy",
        "pandas",
        "pyyaml",
        "statsmodels",
        "tqdm",
    ],
)
