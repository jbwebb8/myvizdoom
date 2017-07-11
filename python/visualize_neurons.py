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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

parser = argparse.ArgumentParser(description='visualize neurons')
parser.add_argument("agent_file_path",
                    help="json file containing agent net and learning args")
parser.add_argument("meta_file_path",
                    help="TF .meta file containing network skeleton")
parser.add_argument("params_file_path",
                    help="TF .data file containing network parameters")
parser.add_argument("config_file_path", 
                    help="config file for scenario")
parser.add_argument("-t", "--test-episodes", type=int, default=100, metavar="",
                    help="episodes to be played (default=100)")
parser.add_argument("-a", "--action-set", default="default", metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default="", metavar="", nargs='*',
                    help="layer output names to probe")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
meta_file_path = args.meta_file_path
params_file_path = args.params_file_path
config_file_path = args.config_file_path
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Doom initialized.")
    return game


# Initialize DoomGame and load netwnvork into Agent instance
game = initialize_vizdoom(config_file_path)
sess = tf.Session()
print("Loading the network from:", meta_file_path)
print("Loading the network weights from:", params_file_path)
# TODO: make action_set not necessary
agent = Agent(game=game, agent_file=agent_file_path, action_set=action_set,
              session=sess, meta_file=meta_file_path, 
              params_file=params_file_path)

# Initialize toolbox
layer_shapes = agent.get_layer_shape(layer_names)
layer_sizes = np.ones(len(layer_shapes), dtype=np.int64)
for i in range(len(layer_shapes)):
    for j in range(len(layer_shapes[i])):
        if layer_shapes[i][j] is not None:
            layer_sizes[i] *= layer_shapes[i][j] 
toolbox = Toolbox(layer_sizes=layer_sizes, 
                  state_shape=agent.state.shape)

# Create grid for display
fig = plt.figure()
outer = gridspec.GridSpec(2, 1)
subs = []
dims = [2, len(layer_shapes) + 1]
for i in range(2):
    subs.append(gridspec.GridSpecFromSubplotSpec(1, dims[i], subplot_spec=outer[i]))
plt.show()
