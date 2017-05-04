#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# Adapted from learning_theano.py (credit: ViZDoom authors)
# 
# Uses DQN consisting of an input layer, two convolutional layers,
# one fully connected layer, and an output layer. Learning occurs
# via exploration from an epsilon-greedy policy and sampling random
# minibatches from stored memories.
#####################################################################

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import pickle
import argparse
import json
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

# TODO: could call action class to ensure args make sense (e.g. json file)
# Command line arguments
parser = argparse.ArgumentParser(description='train an agent')
parser.add_argument("agent_file_path",
                    help="json file containing agent net and learning args")
parser.add_argument("config_file_path", help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="number of epochs to train")
parser.add_argument("-s", "--learning-steps", type=int, default=2000,
                    help="learning steps per epoch")
parser.add_argument("-t", "--test-episodes", type=int, default=100,
                    help="test episodes per epoch")
parser.add_argument("-f", "--save-freq", type=int, default=0,
                    help="save params every x epochs")
parser.add_argument("-w", "--watch-episodes", action="store_true", default=False,
                    help="watch episodes after training")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
if not agent_file_path.lower().endswith(".json"): 
    raise Exception("No agent JSON file.")
agent = json.loads(open(agent_file_path).read())
agent_name = agent["network_args"]["name"]
agent_type = agent["network_args"]["type"]
alpha = agent["network_args"]["alpha"]
gamma = agent["network_args"]["gamma"]
epsilon_start = agent["learning_args"]["epsilon_start"]
epsilon_end = agent["learning_args"]["epsilon_end"]
epsilon_const_epochs = agent["learning_args"]["epsilon_const_epochs"]
epsilon_decay_epochs = agent["learning_args"]["epsilon_decay_epochs"]
batch_size = agent["learning_args"]["batch_size"]
replay_memory_size = agent["memory_args"]["replay_memory_size"]

config_file_path = args.config_file_path
results_directory = args.results_directory
if not results_directory.endswith("/"): 
    results_directory += "/"
try:
    os.makedirs(results_directory)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
epochs = args.epochs
learning_steps_per_epoch = args.learning_steps
test_episodes_per_epoch = args.test_episodes
save_freq = args.save_freq
if save_freq == 0: save_freq = epochs
watch_episodes = args.watch_episodes

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10


# Converts and downsamples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


# Stores and learns from memory
class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, 1, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


# Builds DQN
def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("State")
    a = tensor.vector("Action", dtype="int32")
    q2 = tensor.vector("Q2")
    r = tensor.vector("Reward")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    # Create the input layer of the network.
    dqn = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=s1)

    # Add 2 convolutional layers with ReLu activation
    dqn = Conv2DLayer(dqn, num_filters=8, filter_size=[6, 6],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=3)
    dqn = Conv2DLayer(dqn, num_filters=8, filter_size=[3, 3],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=2)

    # Add a single fully-connected layer.
    dqn = DenseLayer(dqn, num_units=128, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1))

    # Add the output layer (also fully-connected).
    # (no nonlinearity as it is for approximating an arbitrary real function)
    dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

    # Define the loss function
    q = get_output(dqn)
    # target differs from q only for the selected action. The following means:
    # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + gamma * (1 - isterminal) * q2)
    loss = squared_error(q, target_q).mean()

    # Update the parameters according to the computed gradient using RMSProp.
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, agent["network_args"]["alpha"])

    # Compile the theano functions
    print("Compiling the network ...")
    function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print("Network compiled.")

    def simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, 1, resolution[0], resolution[1]]))

    # Returns Theano objects for the net and functions.
    return dqn, function_learn, function_get_q_values, simple_get_best_action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        # the value of q2 is ignored in learn if s2 is terminal
        learn(s1, q2, a, r, isterminal)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_const_epochs = 0.1 * epochs  # 10% of learning time
        epsilon_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < epsilon_const_epochs:
            return epsilon_start
        elif epoch < epsilon_decay_epochs:
            # Linear decay
            return epsilon_start - (epoch - epsilon_const_epochs) / \
                               (epsilon_decay_epochs - epsilon_const_epochs) * (epsilon_start - epsilon_end)
        else:
            return epsilon_end

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability epsilon make a random action.
    epsilon = exploration_rate(epoch)
    if random() <= epsilon:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = initialize_vizdoom(config_file_path)

# Action = which buttons are pressed
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)

# Create dqn
net, learn, get_q_values, get_best_action = create_network(len(actions))

print("Starting the training!")

# Train and test agent for specified number of epochs
time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []

    # Training
    print("Training...")
    game.new_episode()
    for learning_step in trange(learning_steps_per_epoch):
        perform_learning_step(epoch)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1
    print("%d training episodes played." % train_episodes_finished)
    train_scores = np.array(train_scores)
    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
    
    # Testing
    print("\nTesting...")
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)
    test_scores = np.array(test_scores)
    print("Results: mean: %.1f±%.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

    # Save network params after specified number of epochs; otherwise store temporarily after each epoch
    if epoch + 1 == epochs:
        results_file_path = results_directory + "weights_final.dump"
        print("Saving network weights in:", results_file_path)
        pickle.dump(get_all_param_values(net), open(results_file_path, "wb"))
    elif (epoch + 1) % save_freq == 0:
        results_file_path = results_directory + "weights_epoch" + str(epoch+1) + ".dump"
        print("Saving network weights in:", results_file_path)
        pickle.dump(get_all_param_values(net), open(results_file_path, "wb"))
    else:
        results_file_path = results_directory + "weights.dump"
        print("Stashing network weights in:", results_file_path)
        pickle.dump(get_all_param_values(net), open(results_file_path, "wb"))

    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

game.close()
print("======================================")

# Watch newly trained agent play episodes
if watch_episodes:
    print("Loading the network weights from:", results_file_path)
    print("Training finished. It's time to watch!")

    # Load the network's parameters from a file
    params = pickle.load(open(results_file_path, "rb"))
    set_all_param_values(net, params)

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)