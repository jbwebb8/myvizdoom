#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
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