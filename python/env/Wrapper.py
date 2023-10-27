from abc import ABC, abstractmethod

class Game(ABC):
    """
    Base class for environment wrappers to work with preconfigured agents. Will add to as needed.
    """

    def __init__(self, 
                 env,
                 episode_timeout=100, 
                 **kwargs):
        # Initialize abstract class
        super().__init__()

        # Env settings
        self.env = env

        # Game settings
        self.episode_timeout = episode_timeout

        # Episode variables
        self.episode_time = 0
    
    def load_config(self, config_file):
        """Base method for configuring environment. Passes readable form of
        config file to _add_config method for configuring environment."""
        config = {}
        with open(config_file, 'r') as f:
            for line in f:
                line = ''.join(line.split()) # remove whitespace, tabs, newlines
                if line.count('=') != 1:
                    raise SyntaxError("Improper syntax in config file: %s" % line)
                [k, v] = line.split('=')
                self._add_config(k, v)
    
    def get_available_game_variables(self):
        """Returns existing game variables in environment"""
        return []
    
    def get_game_variable(self, *gvs):
        """Returns values of available game variables in environment"""
        gvs_ = []
        for gv in gvs:
            #gvs_.append(None)
            gvs_.append(self._get_game_variable(gv))
        return gvs_

    def get_state(self):
        return self.game_state

    def get_episode_start_time(self):
        return 0.0

    def get_episode_time(self):
        return self.episode_time
    
    def get_episode_timeout(self):
        return self.episode_timeout

    @abstractmethod
    def init(self):
        """Creates instance of configured environment."""
        pass

    @abstractmethod
    def _add_config(self, k, v):
        """Configures environment based on key-value pair."""
        pass
    
    @abstractmethod
    def new_episode(self):
        """Starts a new episode in the environment."""
        pass
    
    @abstractmethod
    def is_episode_finished(self):
        """Returns true if current episode is finished."""
        pass
    
    @abstractmethod
    def get_available_buttons(self):
        """Returns list of available action indices"""
        pass

    @abstractmethod
    def get_available_buttons_size(self):
        """Returns number of actions"""
        pass

    @abstractmethod
    def get_screen_channels(self):
        """Returns number of channels in screen"""
        pass

    @abstractmethod
    def _get_game_variable(self, gv_key):
        """Return value of game variable in environment"""
        pass
    
    @abstractmethod
    def make_action(self, action, frame_repeat=1):
        """Make action that is repeated frame_repeat times"""
        pass
    
    @abstractmethod
    def get_last_action(self):
        """Returns index of last action taken by agent."""
        pass
    
    @abstractmethod
    def get_total_reward(self):
        """Returns total reward collected in episode."""
        pass

    @abstractmethod
    def close(self):
        """Closes environment instance."""
        pass


class GameState(ABC):
    """Wrapper class for ViZDoom GameState"""

    def __init__(self, env, **kwargs):
        self.env = env

    @property
    @abstractmethod
    def screen_buffer(self):
        return self.env.render()
    
    @property
    @abstractmethod
    def game_variables(self):
        return []