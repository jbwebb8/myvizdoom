#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# Adapted from learning_theano.py (credit: ViZDoom authors)
#####################################################################

from vizdoom import *
from Agent import Agent
from Toolbox import Toolbox
import numpy as np
import tensorflow as tf
import argparse
import os, errno
from time import time, sleep
from tqdm import trange

# Command line arguments
parser = argparse.ArgumentParser(description='Test a trained agent.')
parser.add_argument("agent_file_path",
                    help="json file containing agent net and learning args")
parser.add_argument("meta_file_path",
                    help="TF .meta file containing network skeleton")
parser.add_argument("params_file_path",
                    help="TF .data file containing network parameters")
parser.add_argument("config_file_path", 
                    help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-t", "--test-episodes", type=int, default=100, metavar="",
                    help="episodes to be played (default=100)")
parser.add_argument("-a", "--action-set", default="default", metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default=None, metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", default=4, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
meta_file_path = args.meta_file_path
params_file_path = args.params_file_path
config_file_path = args.config_file_path
results_directory = args.results_directory
if not results_directory.endswith("/"): 
    results_directory += "/"
try:
    os.makedirs(results_directory)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names
max_samples = args.max_samples
trackable = args.track

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Doom initialized.")
    return game


# Initialize DoomGame and load network into Agent instance
game = initialize_vizdoom(config_file_path)
sess = tf.Session()
print("Loading the network from:", meta_file_path)
print("Loading the network weights from:", params_file_path)
# TODO: make action_set not necessary
agent = Agent(game=game, agent_file=agent_file_path, action_set=action_set,
              session=sess, meta_file=meta_file_path, 
              params_file=params_file_path)

# Save action indices
if trackable:
    np.savetxt(results_directory + "action_indices.txt", agent.action_indices)

# TODO: incorporate tracking of max activation data
layer_sizes = agent.get_layer_output(layer_names)
toolbox = []
for i in range(len(layer_names)):
    print(layer_sizes[i].size)
    toolbox.append(Toolbox(layer_sizes[i].size, agent.state.shape, max_samples))

print("Let's watch!")

for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    while not game.is_episode_finished():
        agent.make_best_action()
        if trackable:
            agent.track_action()
            agent.track_position()
        if layer_names is not None:
            output = agent.get_layer_output(layer_names)
            for i in range(len(layer_names)):
                toolbox[i].update_max_data(agent.state, 
                                           agent.position_history[-1],
                                           output[i])
    agent.update_score_history()
    
    # Sleep between episodes
    sleep(1.0)
    if trackable:
        np.savetxt(results_directory + "positions_trial" + str(test_episode+1) + ".txt",
                   agent.get_positions())
        np.savetxt(results_directory + "actions_trial" + str(test_episode+1) + ".txt",
                   agent.get_actions())

scores = agent.get_score_history()
np.savetxt(results_directory + "test_scores.txt", scores)
if layer_names is not None:
    np.save(results_directory + "max_values", max_values)
    np.save(results_directory + "max_states", max_states)
    np.save(results_directory + "max_positions", max_positions)