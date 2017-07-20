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
parser.add_argument("params_file_path",
                    help="TF filename (no extension) containing network \
                          parameters")
parser.add_argument("config_file_path", 
                    help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("--meta-file", default=None, metavar="",
                    help="TF .meta file containing network skeleton")
parser.add_argument("-t", "--test-episodes", type=int, default=100, metavar="",
                    help="episodes to be played (default=100)")
parser.add_argument("-a", "--action-set", default="default", metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default=[], metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", default=4, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
parser.add_argument("-v", "--view-data")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
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
meta_file_path = args.meta_file
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names
max_samples = args.max_samples
trackable = args.track

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom... ", end="")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Done.")
    return game


# Initialize DoomGame and load network into Agent instance
game = initialize_vizdoom(config_file_path)
print("Loading agent... ", end="")
# TODO: make action_set not necessary
agent = Agent(game=game, agent_file=agent_file_path, action_set=action_set,
              params_file=params_file_path, output_directory=results_directory,
              train_mode=False)
print("Done.")

# Save action indices
if trackable:
    np.savetxt(results_directory + "action_indices.txt", agent.action_indices)

# Initialize toolbox
print("Initializing toolbox... ", end="")
layer_shapes = agent.get_layer_shape(layer_names)
layer_sizes = np.ones(len(layer_shapes), dtype=np.int64)
for i in range(len(layer_shapes)):
    for j in range(len(layer_shapes[i])):
        if layer_shapes[i][j] is not None:
            layer_sizes[i] *= layer_shapes[i][j] 
toolbox = Toolbox(layer_sizes=layer_sizes, 
                  state_shape=agent.state.shape,
                  num_samples=max_samples)
print("Done.")

print("Let's watch!")
for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    while not game.is_episode_finished():
        current_screen = game.get_state().screen_buffer
        agent.update_state(current_screen)
        agent.make_best_action()
        if trackable:
            agent.track_action()
            agent.track_position()
        if len(layer_names) > 0:
            output = agent.get_layer_output(layer_names)
            toolbox.update_max_data(state=agent.state, 
                                    position=agent.position_history[-1],
                                    layer_values=output)
        print("Game tick %d of max %d in test episode %d of %d.        " 
              % (game.get_episode_time() - game.get_episode_start_time(), 
                 game.get_episode_timeout(),
                 test_episode+1,
                 test_episodes), 
              end='\r')
    agent.update_score_history()
    
    # Sleep between episodes
    sleep(1.0)

scores = np.asarray(agent.score_history)
print("\nSaving results... ", end="")
np.save(results_directory + "test_scores", scores)
if trackable:
    np.save(results_directory + "positions",
            np.asarray(agent.position_history))
    np.save(results_directory + "actions",
            np.asarray(agent.position_history))
if len(layer_names) > 0:
    max_values, max_states, max_positions = toolbox.get_max_data()
    for i in range(len(layer_names)):
        slash = layer_names[i].find("/")
        abbr_name = layer_names[i][0:slash]                    
        np.save(results_directory + "max_values_%s" % abbr_name, 
                max_values[i])
        np.save(results_directory + "max_states_%s" % abbr_name,
                max_states[i])
        np.save(results_directory + "max_positions_%s" % abbr_name,
                max_positions[i])
print("Done.")