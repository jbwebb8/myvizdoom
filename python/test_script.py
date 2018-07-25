# NN packages
import numpy as np
import tensorflow as tf

# Environment setup
import sys
sys.path.insert(0, "../gridworld")
from gridworld import gameEnv

# Network setup
from helper import create_agent, create_network, create_wrapper

# Settings
partial_env = False
env_size = 5

# Create env object and game wrapper
env = gameEnv(partial=partial_env, size=env_size)
game = create_wrapper(env, env_type="gridworld")

# Create agent with game wrapper
agent_file_path = "../agents/dqn_agent.json"
params_file_path = None
action_set = "simple"
results_dir = "../tmp/dump/"
agent = create_agent(agent_file_path,
                     game=game, 
                     params_file=params_file_path,
                     action_set=action_set,
                     output_directory=results_dir)