from env.Wrapper import Game
from env.gridworld import gameEnv
import numpy as np

class Gridworld(Game):
    """
    Wrapper class for Gridworld environment to confer DoomGame functions
    needed by Agent subclasses.
    """

    def __init__(self, 
                 verbose=False, 
                 episode_timeout=100,
                 partial=False,
                 size=5,
                 **kwargs):

        # Game settings
        self.env_config = {'partial': partial,
                           'size': size}
        self.base_config = {'episode_timeout': episode_timeout}
        self.num_channels = 3
        self.verbose = verbose

        # Episode variables
        self.is_terminal = False
        self.last_action = None
        self.total_reward = 0.0
    
    def init(self):
        # Create environment
        args = [self.env_config['partial'], self.env_config['size']]
        env = gameEnv(*args)

        # Initialize base class
        Game.__init__(self, env, env.renderEnv, **self.base_config)

    def _add_config(self, k, v):
        def raise_error():
            raise SyntaxError("Improper value \"%s\" for key \"%s\"" % (v, k))
        
        k = k.lower()
        v = v.lower()

        if k == "partial":
            if v == "true" or v == '1':
                v = True
            elif v == "false" or v == '0':
                v = False
            else:
                raise_error()
            self.env_config["partial"] = v
        
        elif k == "size":
            try:
                self.env_config["sizeX"] = int(v)
                self.env_config["sizeY"] = int(v)
            except ValueError:
                raise_error()
        
        elif k == "episode_timeout":
            try:
                self.base_config["episode_timeout"] = int(v)
            except ValueError:
                raise_error()

        else:
            raise SyntaxError("Unknown key \"%s\"" % k)

    def new_episode(self):
        self.env.reset()
        self.is_terminal = False
        self.last_action = None
        self.total_reward = 0.0
        self.episode_time = 0

    def is_episode_finished(self):
        # Because the gameObject is removed during the checkGoal() call made
        # during env.step() in the terminal state, checkGoal() cannot
        # independently be used to assess if we're in the terminal state.
        #_, is_terminal = self.env.checkGoal()
        #return is_terminal

        return (self.is_terminal or (self.episode_time >= self.episode_timeout))

    def get_available_buttons(self):
        """Returns list of available action meanings"""
        #TODO: write dict for gridworld
        return None

    def get_available_buttons_size(self):
        """Returns number of actions"""

        return self.env.actions

    def get_screen_channels(self):
        """Returns number of channels in screen"""

        return self.num_channels

    def _get_game_variable(self, gv_key):
        # TODO: edit Agent.py to get game variable values by
        # keyword instead of ViZDoom GameVariable type
        #if gv_key == "position_x":
        #    return self.env.objects[0].x
        #elif gv_key == "position_y":
        #    return self.env.objects[0].y
        #else:
        #    return None
        pass
    
    def make_action(self, action, frame_repeat=1):
        """Make action that is repeated frame_repeat times"""
        if isinstance(action, list):
            a = -1 # no-op
            for i, action_ in enumerate(action):
                if action_ == 1:
                    a = i
                    break
        elif isinstance(action, int):
            a = action
        else:
            raise ValueError("Unknown action input", action)
        
        reward = 0.0
        for i in range(frame_repeat):
            s, r, is_terminal = self.env.step(a)
            reward += r
            if is_terminal:
                break
        
        self.is_terminal = is_terminal
        self.last_action = action
        self.total_reward += reward
        self.episode_time += frame_repeat

        return reward
    
    def get_last_action(self):
        return self.last_action
    
    def get_total_reward(self):
        return self.total_reward
    
    def get_episode_time(self):
        return self.episode_time
    
    def close(self):
        self.env = None

class GameVariable:
    """Wrapper class for ViZDoom GameVariables"""

    POSITION_X = "position_x"  
    POSITION_Y = "position_y"


