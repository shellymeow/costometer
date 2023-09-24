from pathlib import Path

import dill as pickle
from mouselab.agents import Agent
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv
from mouselab.policies import RandomPolicy

directory = Path(__file__).parents[0]


def get_q_function(env_name):
    # get environment
    env = MouselabEnv.new_symmetric_registered(env_name)
    # solve for q values
    Q, V, pi, info = timed_solve_env(env, save_q=True)

    # save file
    with open(directory.joinpath(f"{env_name}_q_function.pickle"), "wb") as filename:
        pickle.dump(info["q_dictionary"], filename)


def create_trajectory(env_name, num_episodes=10):
    agent = Agent()

    # get environment
    env = MouselabEnv.new_symmetric_registered(env_name)
    agent.register(env)

    agent.register(RandomPolicy())

    trace = agent.run_many(num_episodes=num_episodes)
    # save file
    filename = open(directory.joinpath(f"{env_name}_trace.pickle"), "wb")
    pickle.dump(trace, filename)
