from vizdoom import *
import numpy as np

class Agent:
    def __init__(self, network, game):
        self.net = network
        self.game = game
        self.positions = []
        self.actions = []
        self.action_indices = self.game.get_available_buttons()

    def _create_network(self):
        # Does agent need access to own network?

    def track_position(self):
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        pos_z = self.game.get_game_variable(GameVariable.POSITION_Z)
        timestamp = self.game.get_episode_time()
        self.positions.append([timestamp, pos_x, pos_y, pos_z])
    
    def track_action(self):
        last_action = self.game.get_last_action()
        timestamp = self.game.get_episode_time()
        self.actions.append([timestamp, last_action])
    
    def get_positions(self):
        return np.asarray(self.positions)
    
    def get_actions(self):
        #action_array = np.empty([len(self.actions), 
        #                        1 + self.game.get_available_buttons_size()],
        #                        dtype=float)
        #for i in range(len(self.actions)):
        #    action_array[i, :] = np.hstack(self.actions[i])
        return np.asarray(self.actions)