from vizdoom import *
import tensorflow as tf
import numpy as np
import argparse
import os, errno, sys
from shutil import copy
from tqdm import trange

# Command line arguments
parser = argparse.ArgumentParser(description="Create a training batch.")
parser.add_argument("config_file_path", help="config file for scenario")
parser.add_argument("results_directory",
                    help="directory where results will be saved")
parser.add_argument("-n", "--num_frames", type=int, default=1000, metavar="",
                    help="number of screen buffer-position pairs to generate")
parser.add_argument("-s", "--skip_rate", type=int, default=4, metavar="",
                    help="number of frames to skip between saved data points")
parser.add_argument("-f", "--save_freq", type=int, default=100, metavar="",
                    help="save backup every x frames")
parser.add_argument("--name", default="", metavar="", 
                    help="experiment name")
parser.add_argument("-d", "--description", default="training batch", metavar="", 
                    help="description of experiment")
args = parser.parse_args()

# Grab arguments
config_file_path = args.config_file_path
results_dir = args.results_directory
num_frames = args.num_frames
skip_rate = args.skip_rate
if not results_dir.endswith("/"): 
    results_dir += "/"
exp_name = args.name
exp_descr = args.description

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
def save_exp_details(folder):
    f = open(folder + "settings.txt", "w+")
    f.write("Name: " + exp_name + "\n")
    f.write("Description: " + exp_descr + "\n")
    f.write("Config file: " + config_file_path + "\n")
    f.write("Number of frames: " + str(num_frames) + "\n")
    f.write("Skip rate: " + str(skip_rate) + "\n")
    files_to_copy = [config_file_path]
    for fp in files_to_copy:
        new_fp = folder + fp.split("/")[-1]
        while os.path.exists(new_fp):
            t = new_fp.split(".")
            new_fp = '.'.join(['.'.join(t[0:-1]) + '_1', t[-1]])
        copy(fp, new_fp)

# Initializes DoomGame from config file
def initialize_vizdoom(config_file):
    game = DoomGame()
    print("Initializing ViZDoom...", end=""), sys.stdout.flush()
    game.load_config(config_file)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    game.init()
    print("Done."), sys.stdout.flush()
    return game

# Set up environment
make_directory([results_dir])
save_exp_details(results_dir)
game = initialize_vizdoom(config_file_path)
screen_buffers = []
positions = []
fn_prefix = results_dir + exp_name + ['' if exp_name == '' else '_'][0]
#image_height = game.get_screen_height()
#image_width = game.get_screen_width()
#print("Initializing buffers..."), sys.stdout.flush()
#screen_buffers = np.zeros([num_frames, 3, image_height, image_width])
#positions = np.zeros([num_frames, 2])
#print("Done."), sys.stdout.flush()
game.new_episode()

# Acquire screen buffer-position pairs
for i in trange(num_frames * skip_rate):
    if game.is_episode_finished():
        game.new_episode()
    if i % skip_rate == 0:
        screen_buffers.append(game.get_state().screen_buffer)
        pos_x = game.get_game_variable(GameVariable.POSITION_X)
        pos_y = game.get_game_variable(GameVariable.POSITION_Y)
        positions.append([pos_x, pos_y])
    game.advance_action()
    #print("Game tick %d of %d" % (i+1, num_frames), end="\r")
    if i / skip_rate % 100 == 0:
        print("Saving...", end=""), sys.stdout.flush()
        np.save(fn_prefix + "screen_buffers", np.asarray(screen_buffers))
        np.save(fn_prefix + "positions", np.asarray(positions))
        print("Done."), sys.stdout.flush()

print("Saving...", end=""), sys.stdout.flush()
np.save(fn_prefix + "screen_buffers", np.asarray(screen_buffers))
np.save(fn_prefix + "positions", np.asarray(positions))
print("Done."), sys.stdout.flush()