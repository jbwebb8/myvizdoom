#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# Adapted from learning_theano.py (credit: ViZDoom authors)
#####################################################################

from vizdoom import *
from agent.Agent import Agent
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
parser.add_argument("-t", "--test-episodes", type=int, default=10, metavar="",
                    help="episodes to be played (default=10)")
parser.add_argument("-a", "--action-set", default="default", metavar="",
                    help="name of action set available to agent")
parser.add_argument("-l", "--layer-names", default="", metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-c", "--color", default="RGB", 
                    choices=["RGB", "RBG", "GBR", "GRB", "BRG", "BGR"],
                    metavar="", help="order of color channels (if color img)")
parser.add_argument("-d", "--discontinuous", action="store_true", default=False,
                    help="pause after each time step for user input")
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
discontinuous = args.discontinuous

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
    
    # Upper row
    inner = gridspec.GridSpecFromSubplotSpec(1, agent.phi+1, 
                                             subplot_spec=outer[0])
    ax = []
    for j in range(agent.phi+1):
        ax_j = plt.Subplot(fig, inner[j])
        fig.add_subplot(ax_j)
        ax_j.axis('off')
        ax.append(ax_j)
    axes.append(ax)

    # Lower row
    inner = gridspec.GridSpecFromSubplotSpec(1, len(layer_names)+1, 
                                             subplot_spec=outer[1])
    ax = []
    for j in range(len(layer_names)):
        ax_j = [] #plt.Subplot(fig, inner[j])
        #fig.add_subplot(ax_j)
        #ax_j.axis('off')
        #print(layer_shapes[j])
        if len(layer_shapes[j]) == 4:
            n = int(np.ceil(np.sqrt(layer_shapes[j][3])))
            grid = gridspec.GridSpecFromSubplotSpec(n, n, 
                                                    subplot_spec=inner[j])
            n_square = int(n*n)
            for k in range(n_square):
                a_k = plt.Subplot(fig, grid[k])
                fig.add_subplot(a_k)
                a_k.axis('off')
                ax_j.append(a_k)
        else:
            ax_j = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax_j)
            ax_j.axis('off')
        ax.append(ax_j)
    axes.append(ax)
    
    # Set axis limits
    axes[0][agent.phi].set_xbound(lower=-600, upper=600)
    axes[0][agent.phi].set_ybound(lower=-600, upper=600)
    axes[0][agent.phi].set_aspect('equal')
    print("Done.")
    return fig, axes

def preprocess_state(state):
    if state.shape[0] == agent.phi * agent.channels:
        state = np.transpose(state, [1, 2, 0])
    imgs = np.split(state, agent.phi, axis=2)
    if agent.channels == 3:
        r = color_order.find("R")
        g = color_order.find("G")
        b = color_order.find("B")
        imgs = [imgs[i][..., [r, g, b]] for i in range(len(imgs))]
    elif agent.channels == 1:
        imgs = [np.squeeze(img) for img in imgs]
    return np.asarray(imgs)

# Initialize DoomGame and load netwnvork into Agent instance
game = initialize_vizdoom(config_file_path)
sess = tf.Session()
print("Loading the network from:", meta_file_path)
print("Loading the network weights from:", params_file_path)
# TODO: make action_set not necessary
agent = Agent(game=game, agent_file=agent_file_path, action_set=action_set,
              session=sess, meta_file=meta_file_path, 
              params_file=params_file_path, output_directory="../tmp/tmp_results/")

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
plt.ion()
#print(agent.game.get_state().screen_buffer)
for test_episode in range(test_episodes):
    agent.initialize_new_episode()
    step = 1
    while not game.is_episode_finished(): 
        # Update state and position
        current_screen = game.get_state().screen_buffer
        agent.update_state(current_screen)
        agent.track_position()

        # Display state
        state = agent.state
        #print(np.max(agent.state))
        images = preprocess_state(state)
        for i in range(agent.phi):
            img = axes[0][i].imshow(images[i])
         
        
        # Display position
        pos = agent.position_history[-1]
        axes[0][agent.phi].plot(pos[1], pos[2], 'x', scalex=False, scaley=False)

        # Display layers
        layer_output = agent.get_layer_output(layer_names)
        for i in range(len(layer_names)):
            if layer_output[i].ndim == 4:
                for j in range(32):
                    axes[1][i][j].imshow(np.squeeze(layer_output[i][..., j]), cmap="gray")
            else:
                #print(layer_output[i].shape)
                axes[1][i].imshow(layer_output[i])
        #q_values = agent.get_layer_output("Q:0")
        #axes[1][len(layer_names)].imshow(q_values)
        
        # Refresh image
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)
        if discontinuous:
            input("Step %d: Press Enter to continue..." % step)
        
        # Make action
        idx = np.random.randint(4)
        game.make_action(agent.actions[idx], 4)
        #agent.make_best_action(train_mode=False)

        step += 1
    fig.clear()
plt.show()
