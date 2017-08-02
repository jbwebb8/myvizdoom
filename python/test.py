#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# Adapted from learning_theano.py (credit: ViZDoom authors)
#####################################################################

from vizdoom import *
from helper import create_agent
from Toolbox import Toolbox
import numpy as np
import tensorflow as tf
import argparse
import warnings
import os, errno
from shutil import copy
from time import time, sleep
from tqdm import trange
import matplotlib.pyplot as plt

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
parser.add_argument("-t", "--test-episodes", type=int, default=100, metavar="",
                    help="episodes to be played (default=100)")
parser.add_argument("-a", "--action-set", default="default", metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default=[], metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", default=0, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("-v", "--visualize-network", action="store_true", default=False,
                    help="visualize agent state and network activation")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
parser.add_argument("-q", "--view-q-values", action="store_true", default=False,
                    help="view real-time Q values")
parser.add_argument("-n", "--name", default="test", metavar="", 
                    help="experiment name (for saving files)")
parser.add_argument("-d", "--description", default="testing", metavar="", 
                    help="description of experiment")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
params_file_path = args.params_file_path
config_file_path = args.config_file_path
results_dir = args.results_directory
if not results_dir.endswith("/"): 
    results_dir += "/"
meta_file_path = args.meta_file
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names
max_samples = args.max_samples
visualize_network = args.visualize_network
trackable = args.track
view_q_values = args.view_q_values
exp_name = args.name
exp_descr = args.description

def make_directory(folders):
    for f in folders:
        try:
            os.makedirs(f)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom... ", end="")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Done.")
    return game

# Make output directories
details_dir = results_dir + "details/"
game_dir = results_dir + "game_data/"
max_dir = results_dir + "max_data/"
make_directory([results_dir, details_dir, game_dir, max_dir])

# Save txt file of important experimental settings
# and copy (small) configuration files
f = open(details_dir + "settings.txt", "w+")
f.write("Name: " + exp_name + "\n")
f.write("Description: " + exp_descr + "\n")
f.write("Agent file: " + agent_file_path + "\n")
f.write("Params file: " + params_file_path + "\n")
f.write("Config file: " + config_file_path + "\n")
f.write("Action set: " + action_set)
files_to_copy = [agent_file_path, config_file_path]
for fp in files_to_copy:
    copy(fp, details_dir)

# Initialize DoomGame and load network into Agent instance
game = initialize_vizdoom(config_file_path)
print("Loading agent... ", end="")
# TODO: make action_set not necessary
agent = create_agent(agent_file_path,
                     game=game,
                     action_set=action_set,
                     params_file=params_file_path, 
                     output_directory=results_dir,
                     train_mode=False)
print("Done.")

# Save action indices
if trackable:
    np.savetxt(game_dir + "action_indices.csv", 
               agent.action_indices,
               delimiter=",",
               fmt="%.0d")

# Initialize toolbox
print("Initializing toolbox... ", end="")
layer_shapes = agent.get_layer_shape(layer_names)
toolbox = Toolbox(layer_shapes=layer_shapes, 
                  state_shape=agent.state.shape,
                  phi=agent.phi,
                  channels=agent.channels,
                  actions=agent.action_indices,
                  num_samples=max_samples)
print("Done.")

# Test agent performance in scenario
print("Let's watch!")
for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    while not game.is_episode_finished():
        # Update state and make best action
        current_screen = game.get_state().screen_buffer
        agent.update_state(current_screen)
        agent.make_best_action()

        # Store and show specified features
        output = None
        if trackable:
            agent.track_action()
            agent.track_position()
        if max_samples > 0:
            output = agent.get_layer_output(layer_names)
            toolbox.update_max_data(state=agent.state, 
                                    position=agent.position_history[-1],
                                    layer_values=output)
        if visualize_network:
            if output == None:
                output = agent.get_layer_output(layer_names)
            toolbox.visualize_features(state=agent.state, 
                                       position=agent.position_history[-1],
                                       layer_values=output)
        if view_q_values:
            q = agent.get_layer_output("Q")
            print(q)
            toolbox.display_q_values(q)   
            sleep(0.5) # HACK: network only works in PLAYER mode, so needed to slow down video

        print("Game tick %d of max %d in test episode %d of %d.        " 
              % (game.get_episode_time() - game.get_episode_start_time(), 
                 game.get_episode_timeout(),
                 test_episode+1,
                 test_episodes), 
              end='\r')
    
    agent.update_score_history()
    sleep(1.0) # sleep between episodes

print("\nSaving results... ", end="")

# Save test scores
scores = np.asarray(agent.score_history)
np.savetxt(game_dir + "test_scores.csv", 
           scores,
           delimiter=",",
           fmt="%.3f")

# Save game traces if tracking specified
if trackable:
    np.savetxt(game_dir + "positions.csv",
               np.asarray(agent.position_history),
               delimiter=",",
               fmt="%.3f")
    np.savetxt(game_dir + "actions.csv",
               np.asarray(agent.action_history),
               delimiter=",",
               fmt="%d")

# Save max data of layers if specified
if len(layer_names) > 0:
    max_values, max_states, max_positions = toolbox.get_max_data()
    for i in range(len(layer_names)):
        slash = layer_names[i].find("/")
        abbr_name = layer_names[i][0:slash]                    
        np.save(max_dir + "max_values_%s" % abbr_name, 
                max_values[i])
        np.save(max_dir + "max_states_%s" % abbr_name,
                max_states[i])
        np.save(max_dir + "max_positions_%s" % abbr_name,
                max_positions[i])

print("Done.")