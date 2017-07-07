###############################################################################
# Name: learn.py
# Description: Train neural network in TensorFlow
###############################################################################

from __future__ import division
from __future__ import print_function
from vizdoom import *
from Agent import Agent
from Toolbox import Toolbox
import numpy as np
import tensorflow as tf
import argparse
import os, errno
from time import time
from tqdm import trange

# Command line arguments
parser = argparse.ArgumentParser(description="Train an agent.")
parser.add_argument("agent_file_path",
                    help="json file containing agent net and learning args")
parser.add_argument("config_file_path", help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-a", "--action-set", default="default",
                    help="name of action set available to agent")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="number of epochs to train")
parser.add_argument("-s", "--learning-steps", type=int, default=2000,
                    help="learning steps per epoch")
parser.add_argument("-t", "--test-episodes", type=int, default=1,
                    help="test episodes per epoch")
parser.add_argument("-f", "--save-freq", type=int, default=0,
                    help="save params every x epochs")
parser.add_argument("-l", "--layer-names", default=[], metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", type=int, default=1, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
parser.add_argument("-n", "--name", default="",
                    help="experiment name (for saving files)")
parser.add_argument("-v", "--verbose", type=bool, default=False,
                    help="print extra info about network (helpful for \
                    debugging)")
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
action_set = args.action_set
epochs = args.epochs
learning_steps_per_epoch = args.learning_steps
test_episodes_per_epoch = args.test_episodes
save_freq = args.save_freq
if save_freq == 0: save_freq = epochs
layer_names = args.layer_names
max_samples = args.max_samples
trackable = args.track
verbose = args.verbose
exp_name = args.name

# Other parameters
#frame_repeat = 12       # frames to repeat action before choosing again 

def initialize_vizdoom(config_file):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file)
    game.init()
    print("Doom initialized.")
    return game

# Initialize agent and TensorFlow graph
game = initialize_vizdoom(config_file_path)
sess = tf.Session()
agent = Agent(game=game, agent_file=agent_file_path, action_set=action_set,
              session=sess)
init = tf.global_variables_initializer()
sess.run(init)

if trackable:
    np.savetxt(results_directory + "action_indices.txt", agent.action_indices)

# Initialize toolbox
layer_shapes = agent.get_layer_shape(layer_names)
layer_sizes = np.ones(len(layer_shapes), dtype=np.int64)
for i in range(len(layer_shapes)):
    for j in range(len(layer_shapes[i])):
        if layer_shapes[i][j] is not None:
            layer_sizes[i] *= layer_shapes[i][j] 
toolbox = Toolbox(layer_sizes=layer_sizes, 
                  state_shape=agent.state.shape,
                  num_samples=max_samples)

# Train and test agent for specified number of epochs
print("Starting the training!")
test_scores_all = []
time_start = time()
for epoch in range(epochs):
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
    agent.reset_history()
    save_epoch = (epoch + 1 == epochs or (epoch + 1) % save_freq == 0)
    for test_episode in trange(test_episodes_per_epoch):
        agent.initialize_new_episode()
        while not game.is_episode_finished():
            print("Game tick %d of max %d" 
                  % (game.get_episode_time() - game.get_episode_start_time(), 
                     game.get_episode_timeout()), 
                  end='\r')
            agent.make_best_action()
            if save_epoch:
                agent.track_action()
                agent.track_position()
                if len(layer_names) > 0:
                    output = agent.get_layer_output(layer_names)
                    toolbox.update_max_data(state=agent.state, 
                                            position=agent.position_history[-1],
                                            layer_values=output)
        agent.update_score_history()
    
    # Get test results
    test_scores = agent.get_score_history()
    test_scores_all.append([np.mean(test_scores, axis=0), 
                            np.std(test_scores, axis=0)])                      
    print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()),
          "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
    
    # Save results after specified number of epochs; 
    # otherwise store temporarily after each epoch
    if save_epoch:
        results_file_path = results_directory + exp_name + "_model"
        print("Saving network weights in:", results_file_path)
        agent.save_model(results_file_path, global_step=epoch+1, 
                         save_meta=(epoch == 0))
        if trackable:
            print("Saving tracking data in:", results_directory)
            np.save(results_directory + "positions-" + str(epoch+1),
                       agent.get_positions())
            np.save(results_directory + "actions-" + str(epoch+1),
                       agent.get_actions())
        if len(layer_names) > 0:
            print("Saving layer data in:", results_directory)
            max_values, max_states, max_positions = toolbox.get_max_data()
            for i in range(len(layer_names)):
                slash = layer_names[i].find("/")
                abbr_name = layer_names[i][0:slash]                    
                np.save(results_directory + "max_values_%s-%d"
                        % (abbr_name, epoch+1), 
                        max_values[i])
                np.save(results_directory + "max_states_%s-%d"
                        % (abbr_name, epoch+1),
                        max_states[i])
                np.save(results_directory + "max_positions_%s-%d"
                        % (abbr_name, epoch+1),
                        max_positions[i])
    else:
        results_file_path = results_directory + exp_name + "_model"
        print("Stashing network weights in:", results_file_path)
        agent.save_model(results_file_path, global_step=None,
                         save_meta=(epoch == 0))

    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

game.close()
np.save(results_directory + "test_scores", test_scores_all)
print("======================================")