###############################################################################
# Name: learn.py
# Description: Train neural network in TensorFlow
###############################################################################

from __future__ import division
from __future__ import print_function
from vizdoom import *
from helper import create_agent
from Toolbox import Toolbox
import numpy as np
import tensorflow as tf
import argparse
import os, errno, sys
from shutil import copy
from time import time
from tqdm import trange

# Command line arguments
parser = argparse.ArgumentParser(description="Train an agent.")
parser.add_argument("agent_file_path",
                    help="json file containing agent net and learning args")
parser.add_argument("config_file_path", help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-p", "--params-file", default=None, metavar="", 
                    help="TF filename (no extension) containing network \
                          parameters")
parser.add_argument("-a", "--action-set", default="default", metavar="", 
                    help="name of action set available to agent")
parser.add_argument("-e", "--epochs", type=int, default=100, metavar="", 
                    help="number of epochs to train")
parser.add_argument("-s", "--learning-steps", type=int, default=2000, metavar="", 
                    help="learning steps per epoch")
parser.add_argument("-t", "--test-episodes", type=int, default=1, metavar="", 
                    help="test episodes per epoch")
parser.add_argument("-f", "--save-freq", type=int, default=0, metavar="", 
                    help="save params every x epochs")
parser.add_argument("-l", "--layer-names", default=[], metavar="", nargs='*',
                    help="layer output names to probe")
parser.add_argument("-m", "--max-samples", type=int, default=1, metavar="",
                    help="# of samples associated with max node activation")
parser.add_argument("-c", "--color", default="RGB", 
                    choices=["RGB", "RBG", "GBR", "GRB", "BRG", "BGR"],
                    metavar="", help="order of color channels (if color img)")
parser.add_argument("--track", action="store_true", default=False,
                    help="track agent position and action")
parser.add_argument("-g", "--save-gifs", default=None,
                    choices=[None, "game_screen", "agent_state"],
                    help="make gifs of agent test episodes with specified images")
parser.add_argument("-n", "--name", default="train", metavar="", 
                    help="experiment name (for saving files)")
parser.add_argument("-d", "--description", default="training", metavar="", 
                    help="description of experiment")
args = parser.parse_args()

# Grab arguments from agent file and command line args
agent_file_path = args.agent_file_path
if not agent_file_path.lower().endswith(".json"): 
    raise Exception("No agent JSON file.")
config_file_path = args.config_file_path
results_dir = args.results_directory
if not results_dir.endswith("/"): 
    results_dir += "/"
# TODO: check for accidental overwrite
#if len(os.listdir(results_dir)) > 0:
#    pass
params_file_path = args.params_file
action_set = args.action_set
epochs = args.epochs
learning_steps_per_epoch = args.learning_steps
test_episodes_per_epoch = args.test_episodes
save_freq = args.save_freq
if save_freq == 0: save_freq = epochs
layer_names = args.layer_names
max_samples = args.max_samples
color_format = args.color
trackable = args.track
save_gifs = args.save_gifs
exp_name = args.name
exp_descr = args.description

# Other parameters
#frame_repeat = 12       # frames to repeat action before choosing again 

# Makes directory if does not already exist
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
    f.write("Action set: " + action_set + "\n")
    f.write("Epochs: " + str(epochs) + "\n")
    f.write("Learning steps per epoch: " + str(learning_steps_per_epoch) + "\n")
    f.write("Test episodes per epoch: " + str(test_episodes_per_epoch))
    files_to_copy = [agent_file_path, net_file_path, config_file_path]
    for fp in files_to_copy:
        new_fp = folder + fp.split("/")[-1]
        while os.path.exists(new_fp):
            t = new_fp.split(".")
            new_fp = '.'.join(['.'.join(t[0:-1]) + '_1', t[-1]])
        copy(fp, new_fp)

# Initializes DoomGame from config file
def initialize_vizdoom(config_file):
    print("Initializing doom... ", end=""), sys.stdout.flush()
    game = DoomGame()
    game.load_config(config_file)
    game.init()
    print("Done.")
    return game  

# Make output directories
details_dir = results_dir + "details/"
game_dir = results_dir + "game_data/"
max_dir = results_dir + "max_data/"
make_directory([results_dir, details_dir, game_dir, max_dir])

# Initialize agent and TensorFlow graph
game = initialize_vizdoom(config_file_path)
print("Loading agent... ", end=""), sys.stdout.flush()
agent = create_agent(agent_file_path,
                     game=game, 
                     params_file=params_file_path,
                     action_set=action_set,
                     output_directory=results_dir)
if trackable:
    np.savetxt(game_dir + "action_indices.txt", 
               agent.action_indices)
print("Done.")

# Initialize toolbox
print("Initializing toolbox... ", end=""), sys.stdout.flush()
layer_shapes = agent.get_layer_shape(layer_names)
toolbox = Toolbox(layer_shapes=layer_shapes, 
                  state_shape=agent.state.shape,
                  phi=agent.phi,
                  channels=agent.channels,
                  actions=agent.action_indices,
                  num_samples=max_samples,
                  data_format=agent.network.data_format,
                  color_format=color_format)
print("Done.")

# Save experimental details
save_exp_details(details_dir, agent)

# Train and test agent for specified number of epochs
print("Starting the training!")
test_scores_all = []
train_scores_all = []
time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0

    # Training
    print("Training...")
    agent.set_train_mode(True)
    agent.reset_history()
    agent.initialize_new_episode()
    for learning_step in trange(learning_steps_per_epoch):
        agent.perform_learning_step(epoch, epochs)
        if game.is_episode_finished():
            agent.update_score_history()
            agent.initialize_new_episode()
            train_episodes_finished += 1
    print("%d training episodes played." % train_episodes_finished)
    train_scores = np.asarray(agent.score_history)
    train_scores_all.append([np.mean(train_scores, axis=0), 
                             np.std(train_scores, axis=0)])
    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
    
    # Testing
    print("\nTesting...")
    agent.set_train_mode(False)
    agent.reset_history()
    save_epoch = (epoch + 1 == epochs or (epoch + 1) % save_freq == 0)
    for test_episode in range(test_episodes_per_epoch):
        agent.initialize_new_episode()
        screen_history = []
        while not game.is_episode_finished():
            current_screen = game.get_state().screen_buffer
            agent.update_state(current_screen)
            if save_epoch:
                if trackable:
                    agent.track_action()
                    agent.track_position()
                if save_gifs is not None:
                    if save_gifs == "agent_state":
                        current_screen = agent._preprocess_image(current_screen)
                    screen_history.append(current_screen)
            agent.make_action()
            print("Game tick %d of max %d in test episode %d of %d." 
                  % (game.get_episode_time() - game.get_episode_start_time(), 
                     game.get_episode_timeout(),
                     test_episode+1,
                     test_episodes_per_epoch), 
                  end='\r')
        
        # Update score history and save gifs (too much overhead to save all at end)
        agent.update_score_history()
        if save_epoch and save_gifs is not None:
            gif_file_path = game_dir + "test_episode%d-%d" \
                            % (epoch+1, test_episode+1)
            toolbox.make_gif(screen_history, game_dir)
    
    # Get test results
    test_scores = np.asarray(agent.score_history)
    test_scores_all.append([np.mean(test_scores, axis=0), 
                            np.std(test_scores, axis=0)])                      
    print("\r\x1b[K" + "Results: mean: %.1f±%.1f," 
          % (test_scores.mean(), test_scores.std()),
          "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())
    
    # Save results after specified number of epochs; 
    # otherwise store temporarily after each epoch
    if save_epoch:
        model_filename = exp_name + "_model"
        print("Saving network... ", end="")
        agent.save_model(model_filename, global_step=epoch+1, 
                         save_meta=(epoch == 0), save_summaries=True)
        if trackable:
            sfx = str(epoch+1) + ".csv"
            np.savetxt(game_dir + "positions-" + sfx,
                       np.asarray(agent.position_history),
                       delimiter=",",
                       fmt="%.3f")
            np.savetxt(game_dir + "actions-" + sfx,
                       np.asarray(agent.action_history),
                       delimiter=",",
                       fmt="%d")
    else:
        model_filename = exp_name + "_model"
        print("Stashing network... ", end="")
        agent.save_model(model_filename, global_step=None,
                         save_meta=(epoch == 0), save_summaries=False)

    print("Done.")
    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

# Close game and save all scores per epoch
game.close()
np.savetxt(game_dir + "train_scores.csv", 
           np.asarray(train_scores_all),
           delimiter=",",
           fmt="%.3f")
np.savetxt(game_dir + "test_scores.csv", 
           np.asarray(test_scores_all),
           delimiter=",",
           fmt="%.3f")
print("======================================")