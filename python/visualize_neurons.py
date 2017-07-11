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
parser.add_argument("-c", "--color", default="RGB", 
                    choices=["RGB", "RBG", "GBR", "GRB", "BRG", "BGR"],
                    metavar="", help="order of color channels (if color img)")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
meta_file_path = args.meta_file_path
params_file_path = args.params_file_path
config_file_path = args.config_file_path
test_episodes = args.test_episodes
action_set = args.action_set
layer_names = args.layer_names
color_order = args.color

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Doom initialized.")
    return game

def initialize_display():
    print("Initializing display...",)
    fig = plt.figure()
    outer = gridspec.GridSpec(2, 1)
    axes = []
    dims = [agent.phi + 1, len(layer_shapes) + 1]
    for i in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(1, dims[i], subplot_spec=outer[i])
        ax = []
        for j in range(dims[i]):
            ax.append(plt.Subplot(fig, inner[j]))
        axes.append(ax)
    print("Done.")
    return fig, axes

def preprocess_state(state):
    if state.shape[0] == agent.phi * agent.channels:
        state = np.transpose(state, [1, 2, 0])
    imgs = np.split(state, agent.phi, axis=2)
    r = color_order.find("R")
    g = color_order.find("G")
    b = color_order.find("B")
    return [np.transpose(imgs[i], [r, g, b]) for i in range(len(imgs))]

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

# Initialize plot
fig, axes = initialize_display()
#print(agent.game.get_state().screen_buffer)
for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    while not game.is_episode_finished():
        # Display state
        state = agent.state
        #print(np.max(agent.state))
        images = preprocess_state(state)
        for i in range(agent.phi):
            #print(images[i])
            
            img = axes[0][i].imshow(images[i])
            fig.add_subplot(axes[0][i])
        plt.show()
        input("Press Enter to continue...")
        # Display position

        # Display layers
        agent.make_best_action(train_mode=False)

plt.show()
