###############################################################################
# Credit: Adapted from spectator.py created by ViZDoom authors at 
# https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/spectator.py
###############################################################################

from vizdoom import *
from time import sleep
import argparse
import sys

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom... ", end=""), sys.stdout.flush()
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    game.init()
    print("Done.")
    return game

# Initialize DoomGame
parser = argparse.ArgumentParser(description="Test a scenario.")
parser.add_argument("config_file_path", 
                    help="config file for scenario")
parser.add_argument("-t", "--test-episodes", type=int, default=10, metavar="",
                    help="episodes to be played (default=10)")
args = parser.parse_args()
config_file_path = args.config_file_path
episodes = args.test_episodes
game = initialize_vizdoom(config_file_path)

for i in range(episodes):
    # Create new episode
    print("Episode #" + str(i + 1))
    game.new_episode()
    frame = 0
    running_r = 0
    while not game.is_episode_finished():
        # Get state, advance current user action, and obtain reward
        state = game.get_state()
        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()
        running_r += reward

        # Print results
        frame += 1
        if frame % 10 == 0:
            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward: %.2f (%.2f total)" % (reward, running_r))
            print("X:", game.get_game_variable(GameVariable.POSITION_X))
            print("=====================")

    print("Episode finished!")
    print("State #" + str(state.number))
    print("Game variables: ", state.game_variables)
    print("Action:", last_action)
    print("Last reward: %.2f" % reward)
    print("X:", game.get_game_variable(GameVariable.POSITION_X))
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
