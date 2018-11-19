# Create wrapper for Atari2600 via OpenAI Gym Environment
import gym
from env.Wrapper import Game, GameState
import numpy as np

# Available Atari enviroments
ATARI_BINS = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 
              'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 
              'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 
              'carnival', 'centipede', 'chopper_command', 'crazy_climber', 
              'defender', 'demon_attack', 'double_dunk', 'elevator_action', 
              'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 
              'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 
              'kaboom', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 
              'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 
              'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 
              'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', 
              'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 
              'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

ATARI_ENVS = []
for game in ATARI_BINS:
    #### Source: gym/__init__.py ###
    for obs_type in ['image', 'ram']:
        # Name format: GameName-(ram)-v#
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        ATARI_ENVS.append('{}-v0'.format(name))
        ATARI_ENVS.append('{}-v4'.format(name))    
        ATARI_ENVS.append('{}Deterministic-v0'.format(name))
        ATARI_ENVS.append('{}Deterministic-v4'.format(name))
        ATARI_ENVS.append('{}NoFrameskip-v0'.format(name))
        ATARI_ENVS.append('{}NoFrameskip-v4'.format(name))

class Atari(Game):
    """
    Wrapper class for Gridworld environment to confer DoomGame functions
    needed by Agent subclasses.
    """

    def __init__(self,
                 env_name='SpaceInvaders-v0',
                 verbose=False, 
                 episode_timeout=100,
                 **kwargs):

        # Game settings
        self.base_config = {'episode_timeout': episode_timeout}
        self.env_config = {'id': env_name}
        self.state_config = {'render_mode': 'rgb_array'}
        self.verbose = verbose

        # Episode variables
        self.is_terminal = False
        self.last_action = None
        self.total_reward = 0.0
    
    def init(self):
        # Create environment
        env = gym.make(**self.env_config)

        # Initialize base class
        Game.__init__(self, env, **self.base_config) 

        # Initialize game state
        self.game_state = AtariGameState(env, **self.state_config)
        self.num_channels = self.game_state.num_channels

    def _add_config(self, k, v):
        """Configures environment based on key-value pair."""
        
        def raise_error():
            raise SyntaxError("Improper value \"%s\" for key \"%s\"" % (v, k))

        if k == "env_name":
            self.env_config['id'] = v

        elif k == "screen_format":
            if v.lower() == 'rgb':
                self.state_config['screen_format'] = 'rgb'
            elif v.lower() == 'gray':
                self.state_config['screen_format'] = 'gray'
            else:
                raise_error()
        
        elif k == "window_visible":
            if v.lower() == 'true':
                self.state_config['render_mode'] = 'human'
            elif v.lower() == 'false':
                self.state_config['render_mode'] = 'rgb_array'
            else:
                raise_error()

        elif k == "episode_timeout":
            try:
                self.base_config["episode_timeout"] = int(v)
            except ValueError:
                raise_error()

        else:
            raise SyntaxError("Unknown key \"%s\"" % k)
        
    def new_episode(self):
        """Starts a new episode in the environment."""
        self.env.reset()
        self.is_terminal = False
        self.last_action = None
        self.total_reward = 0.0
        self.episode_time = 0

    def is_episode_finished(self):
        """Returns True if episode is done."""
        return (self.is_terminal or (self.episode_time >= self.episode_timeout))

    def get_available_buttons(self):
        """Returns list of available action indices"""
        return self.env.env.get_action_meanings()

    def get_available_buttons_size(self):
        """Returns number of actions"""
        
        return self.env.action_space.n

    def get_screen_channels(self):
        """Returns number of channels in screen"""
        
        return self.num_channels

    def _get_game_variable(self, gv_key):
        """Return value of game variable in environment"""
        
        #if gv_key == "ale_lives":
        #    return self.env.ale.lives()
        #else:
        #    return None
        pass
    
    def make_action(self, action, frame_repeat=1):
        """Make action that is repeated frame_repeat times"""
        
        if isinstance(action, list):
            a = None
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
            s, r, is_terminal, info = self.env.step(a)
            reward += r
            if is_terminal:
                break
        
        self.is_terminal = is_terminal
        self.last_action = action
        self.total_reward += reward
        self.episode_time += frame_repeat

        return reward

    def get_last_action(self):
        """Returns last action taken by agent."""

        return self.last_action
    
    def get_total_reward(self):
        """Returns total reward collected in episode."""
        
        return self.total_reward

    def close(self):
        """Closes environment instance."""
        
        self.env.close()


class AtariGameState(GameState):
    """Wrapper class for ViZDoom GameState"""

    def __init__(self,
                 env, 
                 screen_format='rgb', 
                 render_mode='rgb_array'):
        GameState.__init__(self, env)
        self.screen_format = screen_format
        self.render_mode = render_mode
        self.rgb_to_gray = np.array([0.299, 0.587, 0.114])

    @property
    def screen_buffer(self):            
        if self.render_mode == 'human':
            self.env.render(mode='human')
        return self._get_screen_buffer()
    
    def _get_screen_buffer(self):
        rgb = self.env.render(mode='rgb_array')
        if self.screen_format == 'rgb':
            return rgb
        else:
            gray = np.dot(rgb, self.rgb_to_gray)
            return gray

    @property
    def num_channels(self):
        if self.screen_format == 'rgb':
            return 3
        elif self.screen_format == 'gray':
            return 1

    @property
    def game_variables(self):
        return []

    