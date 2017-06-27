###############################################################################
# Name: learn.py
# Description: Train neural network in TensorFlow
###############################################################################

# TODO: Implement RL algorithm in TensorFlow

from __future__ import division
from __future__ import print_function
from vizdoom import *
import argparse
import os, errno, warnings
from Agent import Agent
import itertools as it
from time import time, sleep
import numpy as np
import tensorflow as tf
from tqdm import trange

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
#parser.add_argument("-w", "--watch-episodes", action="store_true", default=False,
#                    help="watch episodes after training")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
if not agent_file_path.lower().endswith(".json"): 
    raise Exception("No agent JSON file.")
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
#watch_episodes = args.watch_episodes

# Other parameters
#phi = 4                 # stacked input frames
#frame_repeat = 12       # frames to repeat action before choosing again 
#resolution = (30, 45)   # screen resolution of input to network
#episodes_to_watch = 10

def initialize_vizdoom(config_file):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file)
    game.init()
    print("Doom initialized.")
    return game

game = initialize_vizdoom(config_file_path)
sess = tf.Session()
agent = Agent(game=game, agent_file=agent_file_path, session=sess)

print("Starting the training!")

# Train and test agent for specified number of epochs
time_start = time()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(epochs):
    # Initialize variables
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0

    # Training
    print("Training...")
    agent.initialize_new_episode()
    for learning_step in trange(learning_steps_per_epoch):
        agent.perform_learning_step(epoch, epochs)
        if game.is_episode_finished():
            agent.update_score_history()
            agent.initialize_new_episode()
            train_episodes_finished += 1
    print("%d training episodes played." % train_episodes_finished)
    train_scores = agent.get_score_history()
    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
    
    # Testing
    print("\nTesting...")
    agent.reset_score_history()
    for test_episode in trange(test_episodes_per_epoch):
        agent.initialize_new_episode()
        while not game.is_episode_finished():
            agent.make_best_action()
        agent.update_score_history()
    test_scores = agent.get_score_history()
    print("Results: mean: %.1f±%.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
    
    # TODO: implement Saver object
    # Save network params after specified number of epochs; otherwise store temporarily after each epoch
    """
    params = agent.get_network_params()
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
    """
game.close()
print("======================================")