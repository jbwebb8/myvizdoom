#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# Adapted from learning_theano.py (credit: ViZDoom authors)
# 
# Recreates DQN consisting of an input layer, two convolutional layers,
# one fully connected layer, and an output layer. Loads parameters
# from previously trained network (e.g. learning_theano_train.py) and
# allows user to watch footage of bot in action.
#####################################################################

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import pickle
import argparse
import warnings
import os, errno
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange
from Agent import Agent

# Command line arguments
parser = argparse.ArgumentParser(description='Test a trained agent.')
parser.add_argument("params_file_path",
                    help="file containing network parameters")
parser.add_argument("config_file_path", help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-t", "--test-episodes", type=int, default=100, metavar="",
                    help="episodes to be played")
args = parser.parse_args()

# Grab arguments from agent file and command line args
config_file_path = args.config_file_path
params_file_path = args.params_file_path
results_directory = args.results_directory
if not results_directory.endswith("/"): 
    results_directory += "/"
try:
    os.makedirs(results_directory)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
test_episodes = args.test_episodes

# Q-learning settings
discount_factor = 0.99

# Other parameters
phi = 4 # stacked input frames
frame_repeat = 4
resolution = (60, 108)

# Converts and downsamples the input image
def preprocess(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.transform.resize(img, resolution)
    new_img = new_img[np.newaxis,:,:]
    new_img = new_img.astype(np.float32)
    return new_img

def update_state(state, new_img):
    img = preprocess(new_img)
    new_state = np.delete(state, 0, axis=0)
    new_state = np.append(new_state, img, axis=0)
    return new_state

def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("State")
    a = tensor.vector("Action", dtype="int32")
    q2 = tensor.vector("Q2")
    r = tensor.vector("Reward")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    # Create the input layer of the network.
    dqn = InputLayer(shape=[None, phi, resolution[0], resolution[1]], input_var=s1)

    # Add 2 convolutional layers with ReLu activation
    dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=[4, 4])
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=[2, 2])

    # Add a single fully-connected layer.
    dqn = DenseLayer(dqn, num_units=4608, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1))

    # Add the output layer (also fully-connected).
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

    # Define the loss function
    q = get_output(dqn)
    # target differs from q only for the selected action. The following means:
    # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
    loss = squared_error(q, target_q).mean()

    # Compile the theano functions
    print("Compiling the network ...")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print("Network compiled.")

    def simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, phi, resolution[0], resolution[1]]))

    # Returns Theano objects for the net and functions.
    return dqn, function_get_q_values, simple_get_best_action


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = initialize_vizdoom(config_file_path)

# Action = which buttons are pressed
actions = [[1, 0, 0, 0], # TURN_LEFT
           [0, 1, 0, 0], # TURN_RIGHT
           [0, 0, 1, 0], # MOVE_FORWARD
           [1, 0, 1, 0], # TURN_LEFT + MOVE_FORWARD
           [0, 1, 1, 0], # TURN_RIGHT + MOVE_FORWARD
           [0, 0, 0, 1]] # USE

net, get_q_values, get_best_action = create_network(len(actions))
agent = Agent(net, game)
np.savetxt(results_directory + "action_indices.txt", agent.action_indices)

print("Loading the network weigths from:", params_file_path)
print("Let's watch!")

# Load the network's parameters from a file
params = pickle.load(open(params_file_path, "rb"), encoding="latin1")
set_all_param_values(net, params)

for test_episode in range(test_episodes):
    game.new_episode()
    current_state = np.zeros([phi, resolution[0], resolution[1]],
                                 dtype=np.float32)        
    for init_step in range(phi):
        current_screen = game.get_state().screen_buffer
        current_state[init_step, ...] = preprocess(current_screen)
    while not game.is_episode_finished():
        current_screen = game.get_state().screen_buffer
        current_state = update_state(current_state, current_screen)
        best_action_index = get_best_action(current_state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        print(actions[best_action_index])
        agent.track_action()
        for _ in range(frame_repeat):
            game.advance_action()
            agent.track_position()
        
    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)
    np.savetxt(results_directory + "positions_trial" + str(test_episode+1) + ".txt",
               agent.get_positions())
    np.savetxt(results_directory + "actions_trial" + str(test_episode+1) + ".txt",
               agent.get_actions())