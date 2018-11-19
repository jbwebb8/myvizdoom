# Create wrapper for DoomGame
from vizdoom import DoomGame
from env.Wrapper import Game, GameState
from helper import get_game_variables as get_gvs
import numpy as np

class Doom(Game):
    """
    Wrapper class for DoomGame to confer functions with slight
    modifications to be compatible with Game superclass.
    """

    def __init__(self,
                 verbose=False, 
                 **kwargs):

        self.game = DoomGame()

        # Game settings
        self.verbose = verbose

    def load_config(self, config_file):
        """Loads DoomGame configuration file."""
        return self.game.load_config(config_file)
    
    def get_available_game_variables(self):
        """Returns existing game variables in environment"""
        return self.game.get_available_game_variables()

    def _get_game_variable(self, gv_key):
        """Return value of game variable in environment"""
        return self.game.get_game_variable(get_gvs(gv_key))

    def get_state(self):
        return self.game.get_state()

    def get_episode_start_time(self):
        return self.game.get_episode_start_time()

    def get_episode_time(self):
        return self.game.get_episode_time()
    
    def get_episode_timeout(self):
        return self.game.get_episode_timeout()

    def init(self):
        """Creates instance of configured environment."""
        return self.game.init()

    def _add_config(self, k, v):
        """Configures environment based on key-value pair."""
        pass
    
    def new_episode(self):
        """Starts a new episode in the environment."""
        return self.game.new_episode()
    
    def is_episode_finished(self):
        """Returns true if current episode is finished."""
        return self.game.is_episode_finished()
    
    def get_available_buttons(self):
        """Returns list of available action indices"""
        return self.game.get_available_buttons()

    def get_available_buttons_size(self):
        """Returns number of actions"""
        return self.game.get_available_buttons_size()

    def get_screen_channels(self):
        """Returns number of channels in screen"""
        return self.game.get_screen_channels()
    
    def make_action(self, action, frame_repeat=1):
        """Make action that is repeated frame_repeat times"""
        return self.game.make_action(action, frame_repeat)
    
    def get_last_action(self):
        """Returns index of last action taken by agent."""
        return self.game.get_last_action()
    
    def get_total_reward(self):
        """Returns total reward collected in episode."""
        return self.game.get_total_reward()

    def close(self):
        """Closes environment instance."""
        return self.game.close()