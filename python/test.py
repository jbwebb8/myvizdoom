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
import os, errno, sys
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
parser.add_argument("-t", "--test-episodes", type=int, default=10, metavar="",
                    help="episodes to be played (default=10)")
parser.add_argument("-a", "--action-set", default=None, metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default=[], metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", type=int, default=0, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("-v", "--visualize-network", action="store_true", default=False,
                    help="visualize agent state and network activation")
parser.add_argument("-c", "--color", default="RGB", 
                    choices=["RGB", "RBG", "GBR", "GRB", "BRG", "BGR"],
                    metavar="", help="order of color channels (if color img)")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
parser.add_argument("-g", "--save-gifs", default=[None], nargs="*",
                    choices=[None, "game_screen", "agent_state", "features"],
                    help="make gifs of agent test episodes with specified images")
parser.add_argument("-q", "--view-q-values", action="store_true", default=False,
                    help="view real-time Q values")
parser.add_argument("-n", "--name", default="test", metavar="", 
                    help="experiment name (for saving files)")
parser.add_argument("-d", "--description", default="testing", metavar="", 
                    help="description of experiment")
parser.add_argument("--predict-position", action="store_true", default=False, 
                    help="display predicted position and calculate loss")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
params_file_path = args.params_file_path
config_file_path = args.config_file_path
results_dir = args.results_directory
if not results_dir.endswith("/"): 
    results_dir += "/"
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names
max_samples = args.max_samples
visualize_network = args.visualize_network
color_format = args.color
trackable = args.track
save_gifs = args.save_gifs
if not isinstance(save_gifs, list):
    save_gifs = [save_gifs]
view_q_values = args.view_q_values
exp_name = args.name
exp_descr = args.description
pred_pos = args.predict_position

def make_directory(folders):
    for f in folders:
        try:
            os.makedirs(f)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

# Saves txt file of important experimental settings
# and copies (small) configuration files
def save_exp_details(folder, agent):
    f = open(folder + "settings.txt", "w+")
    f.write("Name: " + exp_name + "\n")
    f.write("Description: " + exp_descr + "\n")
    f.write("Agent file: " + agent_file_path + "\n")
    net_file_path = agent.net_file
    f.write("Network file: " + net_file_path + "\n")
    f.write("Params file: " + str(params_file_path) + "\n")
    f.write("Config file: " + config_file_path + "\n")
    f.write("Action set: " + str(action_set) + "\n")
    files_to_copy = [agent_file_path, net_file_path, config_file_path]
    for fp in files_to_copy:
        new_fp = folder + fp.split("/")[-1]
        while os.path.exists(new_fp):
            t = new_fp.split(".")
            new_fp = '.'.join(['.'.join(t[0:-1]) + '_1', t[-1]])
        copy(fp, new_fp)

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom... ", end=""), sys.stdout.flush()
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

# Initialize DoomGame
game = initialize_vizdoom(config_file_path)

# Load network into Agent instance
print("Loading agent... ", end=""), sys.stdout.flush()
# TODO: make action_set not necessary
agent = create_agent(agent_file_path,
                     game=game,
                     action_set=action_set,
                     params_file=params_file_path, 
                     output_directory=results_dir,
                     train_mode=False)
np.savetxt(game_dir + "action_indices.csv", 
           agent.action_indices,
           delimiter=",",
           fmt="%.0d")
print("Done.")

# Initialize toolbox
print("Initializing toolbox... ", end=""), sys.stdout.flush()
layer_shapes = agent.get_layer_shape(layer_names)
toolbox = Toolbox(layer_shapes=layer_shapes, 
                  state_shape=agent.state[0].shape,
                  phi=agent.phi,
                  channels=agent.channels,
                  actions=agent.action_indices,
                  num_samples=max_samples,
                  data_format=agent.network.data_format,
                  color_format=color_format,
                  view_features=visualize_network,
                  view_Q_values=view_q_values)
print("Done.")

# Save experimental details
save_exp_details(details_dir, agent)

# Test agent performance in scenario
print("Let's watch!")
screen_history_all, feature_history_all, pred_position_history_all, \
    position_history_all, action_history_all, score_history = [], [], [], [], [], []
for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    toolbox.clear_displays()
    screen_history, feature_history, pred_position_history = [], [], []
    while not game.is_episode_finished():
        # Update current state
        current_screen = game.get_state().screen_buffer
        agent.update_state(current_screen)

        # Get action based on learning algorithm
        a = agent.get_action()
        a = agent.check_position_timeout(a)
        agent.track_position()

        # Store and show specified features
        output = None
        if max_samples > 0:
            output = agent.get_layer_output(layer_names)
            toolbox.update_max_data(state=agent.state[0], 
                                    position=agent.position_history[-1],
                                    layer_values=output)
        if visualize_network:
            if output == None:
                output = agent.get_layer_output(layer_names)
            if pred_pos:
                pred_position = np.squeeze(agent.get_layer_output(["POS"])[0])
                pred_position_history.append(pred_position)
            else:
                pred_position = None
            fig = toolbox.visualize_features(state=agent.state[0], 
                                             position=agent.position_history[-1][1:],
                                             layer_values=output,
                                             pred_position=pred_position)
            if "features" in save_gifs:
                # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
                features = np.fromstring(fig.canvas.tostring_rgb(), 
                                         dtype=np.uint8, 
                                         sep='')
                img_shape = fig.canvas.get_width_height()[::-1] + (3,)
                features = features.reshape(img_shape)
                feature_history.append(features)  
        else: # Avoid computing predicted position twice
            if pred_pos:
                pred_position = np.squeeze(agent.get_layer_output(["POS"])[0])
                pred_position_history.append(pred_position)
        if view_q_values:
            q = agent.get_layer_output("Q")[0] # shape [1, 4] --> [4,]
            fig = toolbox.display_q_values(q)   
            sleep(0.5) # HACK: network only works in PLAYER mode,
                       # so needed to slow down video

        # Make action for specified number of frames
        game.set_action(a)
        for _ in range(agent.frame_repeat):
            if [i for i in ["agent_state", "game_screen"] if i in save_gifs]:
                screen_history.append(current_screen)
            game.advance_action()
            if not game.is_episode_finished():
                current_screen = game.get_state().screen_buffer
        agent.track_action()
        print("Game tick %d of max %d in test episode %d of %d.        " 
              % (game.get_episode_time() - game.get_episode_start_time(), 
                 game.get_episode_timeout(),
                 test_episode+1,
                 test_episodes), 
              end='\r')
    
    # Save episode stats and sleep between episodes
    score_history.append(agent.get_score())
    if trackable:
        position_history_all.append(agent.position_history)
        action_history_all.append(agent.action_history)
    if pred_pos:
        pred_position_history_all.append(pred_position_history)
    if [i for i in ["agent_state", "game_screen"] if i in save_gifs]:
        screen_history_all.append(screen_history)
    if "features" in save_gifs:
        feature_history_all.append(feature_history)
    sleep(1.0)

print("\nSaving results... ", end="")

# Save test scores
np.savetxt(game_dir + "test_scores.csv", 
           np.asarray(score_history),
           delimiter=",",
           fmt="%.3f")

# Save game traces if tracking specified
if trackable:
    for i, [ph, ah] in enumerate(zip(position_history_all, action_history_all)):
        np.savetxt(game_dir + "positions-%d.csv" % (i+1),
                   np.asarray(ph),
                   delimiter=",",
                   fmt="%.3f")
        np.savetxt(game_dir + "actions-%d.csv" % (i+1),
                   np.asarray(ah),
                   delimiter=",",
                   fmt="%d")
if pred_pos:
    for i, pph in enumerate(pred_position_history_all):
        np.savetxt(game_dir + "pred_positions-%d.csv" % (i+1),
                   np.asarray(pred_position_history),
                   delimiter=",",
                   fmt="%.3f")

# Save max data of layers if specified
if len(layer_names) > 0:
    max_values, max_states, max_positions = toolbox.get_max_data()
    for i in range(len(layer_names)):
        slash = layer_names[i].find("/")
        if slash > -1:
            abbr_name = layer_names[i][0:slash]
        else:
            abbr_name = layer_names[i]
        np.save(max_dir + "max_values_%s" % abbr_name, 
                max_values[i])
        np.save(max_dir + "max_states_%s" % abbr_name,
                max_states[i])
        np.save(max_dir + "max_positions_%s" % abbr_name,
                max_positions[i])

# Save gifs of agent gameplay
if "game_screen" in save_gifs:
    for i, sh in enumerate(screen_history_all):
        gif_file_path = game_dir + "test_episode-%d-screen" % (i+1)
        toolbox.make_gif(sh, gif_file_path)
if "agent_state" in save_gifs:
    for i, sh in enumerate(screen_history_all):
        gif_file_path = game_dir + "test_episode-%d-state" % (i+1)
        for i in range(len(sh)):
            sh[i] = agent._preprocess_image(sh[i])
        toolbox.make_gif(sh, gif_file_path)
if "features" in save_gifs:
    print("here")
    for i, fh in enumerate(feature_history_all):
        gif_file_path = game_dir + "test_episode-%d-features" % (i+1)
        toolbox.make_gif(fh, gif_file_path, fps=35/agent.frame_repeat)

print("Done.")