from vizdoom import *
import numpy as np

class Agent:
    def __init__(self, network, game):
        self.net = network
        self.game = game
        self.position = np.zeros([1, 2])

    def track_position(self):
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.position = np.vstack((self.position, [pos_x, pos_y]))

