from env.Wrapper import Wrapper
import numpy as np

class Gridworld(Wrapper):
    """
    Wrapper class for Gridworld environment to confer DoomGame functions
    needed by Agent subclasses.
    """

    def __init__(self, env, verbose=False, episode_timeout=100):

        # Initialize base class
        Wrapper.__init__(self)

        # Game settings
        self.env = env
        self.game_state = GameState(env)
        self.num_channels = 3
        self.verbose = verbose
        self.episode_timeout = episode_timeout

        # Episode variables
        self.is_terminal = False
        self.last_action = None
        self.total_reward = 0.0
        self.episode_time = 0

    def load_config(self, config_file):
        config = {}
        with open(config_file, 'r') as f:
            for line in f:
                line = ''.join(line.split()) # remove whitespace, tabs, newlines
                if line.count('=') != 1:
                    raise SyntaxError("Improper syntax in config file: %s" % line)
                [k, v] = line.split('=')
                self._add_config(k, v)
    
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
            self.env.partial = v
        
        elif k == "size":
            try:
                self.env.sizeX = int(v)
                self.env.sizeY = int(v)
            except ValueError:
                raise_error()
        
        elif k == "episode_timeout":
            try:
                self.episode_timeout = int(v)
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
        """Returns list of one-hot vectors encoding actions"""

        actions = []
        for i in range(self.env.actions):
            a = [0] * self.env.actions
            a[i] = 1
            actions.append(a)
        return actions

    def get_available_buttons_size(self):
        """Returns number of actions"""

        return self.env.actions

    def get_screen_channels(self):
        """Returns number of channels in screen"""

        return self.num_channels

    def get_available_game_variables(self):
        
        return []
    
    def get_game_variable(self, *gvs):
        gvs_ = []
        for gv in gvs:
            gvs_.append(None)
            #gvs_.append(self._get_game_variable(gv))
        return gvs_

    def _get_game_variable(self, gv_key):
        
        if gv_key == "position_x":
            return self.env.objects[0].x
        elif gv_key == "position_y":
            return self.env.objects[0].y

    def get_state(self):
        return self.game_state
    
    def get_episode_time(self):
        return 0
    
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

    def get_episode_start_time(self):
        return 0.0
    
    def get_episode_time(self):
        return self.episode_time
    
    def get_episode_timeout(self):
        return self.episode_timeout
    
    def close(self):
        self.env = None



class GameVariable:
    """Wrapper class for ViZDoom GameVariables"""

    POSITION_X = "position_x"  
    POSITION_Y = "position_y"

class GameState:
    """Wrapper class for ViZDoom GameState"""

    def __init__(self, env):
        self.env = env

    @property
    def screen_buffer(self):
        return self.env.renderEnv()
    
    @property
    def game_variables(self):
        return []
