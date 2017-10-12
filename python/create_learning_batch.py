from vizdoom import *
import tensorflow as tf
import numpy as np
from tqdm import trange
import sys

def initialize_vizdoom(config_file):
    game = DoomGame()
    print("Initializing ViZDoom..."), sys.stdout.flush()
    game.load_config(config_file)
    game.set_window_visible(True)
    game.init()
    print("Done."), sys.stdout.flush()
    return game

config_file = "../config/open_field.cfg"
results_dir = "../experiments/tools/learning_batches/batch_1/"
game = initialize_vizdoom(config_file)
num_frames = 1000
frame_skip = 10
image_height = game.get_screen_height()
image_width = game.get_screen_width()
print("Initializing buffers..."), sys.stdout.flush()
#screen_buffers = np.zeros([num_frames, 3, image_height, image_width])
#positions = np.zeros([num_frames, 2])
screen_buffers = []
positions = []
print("Done."), sys.stdout.flush()

game.new_episode()
for i in trange(num_frames * frame_skip):
    if game.is_episode_finished():
        game.new_episode()
    if i % frame_skip == 0:
        screen_buffers.append(game.get_state().screen_buffer)
        pos_x = game.get_game_variable(GameVariable.POSITION_X)
        pos_y = game.get_game_variable(GameVariable.POSITION_Y)
        positions.append([pos_x, pos_y])
    game.advance_action()
    #print("Game tick %d of %d" % (i+1, num_frames), end="\r")
    if i / frame_skip % 100 == 0:
        print("Saving..."), sys.stdout.flush()
        np.save(results_dir + "screen_buffers", np.asarray(screen_buffers))
        np.save(results_dir + "positions", np.asarray(positions))
        print("Done."), sys.stdout.flush()

print("Saving..."), sys.stdout.flush()
np.save(results_dir + "screen_buffers", np.asarray(screen_buffers))
np.save(results_dir + "positions", np.asarray(positions))
print("Done."), sys.stdout.flush()