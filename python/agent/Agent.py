from vizdoom import *
from helper import create_network
from network.Network import Network
from memory.ReplayMemory import ReplayMemory
import numpy as np
import tensorflow as tf
import skimage.color, skimage.transform
from random import randint, random
import json
import warnings
import os, errno
from utils import recursive_dict_search

class Agent:
    """
    Creates an Agent object that oversees network, memory, and learning functions.

    Args:
    - game: DoomGame instance simulating the environment.
    - output_directory: Directory in which results will be saved.
    - agent_file (optional, default: None): JSON file containing agent settings
        (see README).
    - params_file (optional, default: None): File containing weights from 
        previously trained network.
    - train_mode (optional, default: True): Boolean; True if training agent.
        Important if using batch norm in network to flag use of batch vs.
        population statistics.
    - action_set (optional, default: default): List of actions available for 
        agent. Possible values are:
        - default: [move_forward], [turn_right], [turn_left], [use],
            [move_forward, turn_right], [move_forward, turn_left]
        - basic_four: [move_forward], [turn_right], [turn_left], [use]
        - basic_three: [move_forward], [turn_right], [turn_left]
    - frame_repeat (optional, default: 4): Number of frames to repeat action 
        for each action chosen.
    - **kwargs (optional): May be used in place of agent file.
        - agent_name (default: default): Name of agent.
        - net_file (default: default): JSON file specifying network 
            architecture.
        - alpha (default: 0.00025): Learning rate.
        - gamma (default: 0.99): Discount factor for value iterative methods.
        - phi (default: 1): Number of previous frames that constitute state of
            agent.
        - channels (default: 1): Number of channels in game screen buffer.
        - position_timeout (default: None): Number of times any given action can
            be repeated before another action is chosen randomly (prevents
            infinite looping for actions that do not change state).
        - reward_scale (default: 1.0): Scales rewards built into scenarios by a
            constant.
        - shape_reward_fns (default: None): If True, the reward received
            after the agent makes an action includes a shaping reward encoded
            in the scenario script (under USER1).
    """

    NET_JSON_DIR = "../networks/"
    MAIN_SCOPE = "main_network"
    DEFAULT_AGENT_ARGS = {"agent_name":         "default",
                          "net_file":           "default",
                          "alpha":              0.00025,
                          "gamma":              0.99,
                          "phi":                1,
                          "channels":           1,
                          "frame_repeat":       4,
                          "position_timeout":   None,
                          "no_op":              False,
                          "reward_scale":       1.0,
                          "shape_reward_fns":   None}

    def __init__(self, game, output_directory, agent_file=None,
                 params_file=None, train_mode=True, action_set="default",
                 **kwargs):
        # Initialize game
        self.game = game
        self.sess = tf.Session()
        self.global_step = 0
        self.train_mode = train_mode

        # Set up results directories
        if not output_directory.endswith("/"): 
            output_directory += "/"
        self.net_dir = output_directory + "net_data/"
        self.main_net_dir = self.net_dir + "main_net/"
        self._make_directory([self.net_dir, self.main_net_dir])

        # Initialize action space and gameplay
        self.action_indices = np.asarray(self.game.get_available_buttons())
        self.actions = self._set_actions(action_set)
        self.num_actions = len(self.actions)        
        self.position_repeat = 0
        
        # Load learning and network parameters
        if agent_file is not None:
            self._load_agent_file(agent_file)
        else:
            self.agent_name         = kwargs.pop("agent_name", 
                                                 self.DEFAULT_AGENT_ARGS["agent_name"])
            self.frame_repeat       = kwargs.pop("frame_repeat",
                                                 self.DEFAULT_AGENT_ARGS["frame_repeat"])
            self.position_timeout   = kwargs.pop("position_timeout",
                                                 self.DEFAULT_AGENT_ARGS["position_timeout"])
            self.no_op              = kwargs.pop("no_op",
                                                 self.DEFAULT_AGENT_ARGS["no_op"])
            self.net_file           = kwargs.pop("net_name", 
                                                 self.DEFAULT_AGENT_ARGS["net_file"])
            self.alpha              = kwargs.pop("alpha", 
                                                 self.DEFAULT_AGENT_ARGS["alpha"])
            self.gamma              = kwargs.pop("gamma", 
                                                 self.DEFAULT_AGENT_ARGS["gamma"])
            self.phi                = kwargs.pop("phi",
                                                 self.DEFAULT_AGENT_ARGS["phi"])
            self.channels           = kwargs.pop("channels",
                                                 self.DEFAULT_AGENT_ARGS["channels"])
            self.reward_scale       = kwargs.pop("reward_scale",
                                                 self.DEFAULT_AGENT_ARGS["reward_scale"])
            self.shape_reward_fns   = kwargs.pop("shape_reward_fns",
                                                 self.DEFAULT_AGENT_ARGS["shape_reward_fns"])
            
        if self.channels != self.game.get_screen_channels():
            raise ValueError("Number of image channels between agent and "
                             "game instance do not match. Please check config "
                             "and/or agent file.")
        
        # Save readable network pointers
        if not self.net_file.startswith(self.NET_JSON_DIR):
            self.net_file = self.NET_JSON_DIR + self.net_file
        if not self.net_file.endswith(".json"):
            self.net_file += ".json"
        self.params_file = params_file

        # Create primary network
        self.network = create_network(self.net_file,
                                      phi=self.phi, 
                                      num_channels=self.channels, 
                                      num_outputs=self.num_actions,
                                      learning_rate=self.alpha,
                                      params_file=self.params_file,
                                      output_directory=self.main_net_dir,
                                      session=self.sess,
                                      train_mode=self.train_mode,
                                      scope=self.MAIN_SCOPE)
        
        # Create state = [screen_state, game_variables]
        self.state = []
        self.state.append(np.zeros(self.network.input_shape, dtype=np.float32))
        gvs = self.game.get_available_game_variables()
        self.state.append(np.zeros([len(gvs)], dtype=np.float32))
        self.fixed_to_float_vector = np.ones([len(gvs)], dtype=np.float32)
        for i, gv in enumerate(gvs):
            # Assume USER variables are fixed point numbers
            if str(gv)[-6:-2] == "USER":
                self.fixed_to_float_vector[i] = 1.0 / 2.0**16
        self.num_game_var = len(self.state[1])

        # Cross check agent state with network input states
        print("Mapping of agent states --> network states:")
        print("screen -->", self.network.state[0])
        i = 0
        for j in range(1, len(self.network.state)):
            gv = gvs[i:i+self.network.game_var_sets[j-1]]
            s = self.network.state[j]
            print(gv, "-->", s)
            i += self.network.game_var_sets[j-1]
        if len(gvs) < sum(self.network.game_var_sets):
            raise SyntaxError("Number of inputs in network exceeds number of "
                              + "components in agent state.")
        elif len(gvs) > sum(self.network.game_var_sets):
            extra_gvs = gvs[(len(gvs) - sum(self.network.game_var_sets)):]
            msg = "The following game variables were not used: %s" % ", ".join(map(str, extra_gvs))
            warnings.formatwarning(msg, UserWarning, "", 121)
            warnings.warn(msg)

        # Set up reward shaping function via JSON format if not passed explicitly
        if "shape_reward" not in kwargs:
            self.initialize_shape_reward, self.shape_reward = self._set_shape_reward()

        # Create tracking lists
        self.position_history = []
        self.action_history = []
    
    def _make_directory(self, folders):
        """Makes directories if do not already exist."""
        for f in folders:
            try:
                os.makedirs(f)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

    def _set_actions(self, action_set=None, no_op=True):
        """
        Sets available actions for agent. For dictionary of buttons and their 
        integer values, see ViZDoom/include/ViZDoomTypes.h.

        Args:
        - action_set (optional, default: None): Name of pre-defined list of 
            actions. Possible values are:
            - None: all combinations of available buttons are generated.
            - default: [move_forward], [turn_right], [turn_left], [use],
                [move_forward, turn_right], [move_forward, turn_left]
            - basic_four: [move_forward], [turn_right], [turn_left], [use]
            - basic_three: [move_forward], [turn_right], [turn_left]
        """
        
        def _check_actions(actual_num_buttons, expected_num_buttons):
            if actual_num_buttons < expected_num_buttons:
                raise Exception("Button(s) in action set are not available " 
                                "in game. Please check config file.")
            elif actual_num_buttons < self.game.get_available_buttons_size():
                warnings.warn("Some available game buttons may be unused.")
            else:
                pass

        # If action set is None, create all combinations of available buttons
        # Create choice of simple_combo or all_combo for combination of available buttons in config file
        if action_set is None:
            num_buttons = self.game.get_available_buttons_size()
            actions = []
            for i in range(2 ** num_buttons):
                a = [0] * num_buttons
                j = num_buttons - 1
                while i > 0:
                    if i >= 2 ** j:
                        a[j] = 1
                        i = i - 2 ** j
                    else:
                        j = j - 1
                actions.append(a)
            if not no_op:
                actions.pop(0) # remove no op
            return actions

        # One action for each available button (no combinations)
        if action_set == "simple":
            num_buttons = self.game.get_available_buttons_size()
            actions = []
            for i in range(num_buttons):
                a = [0] * num_buttons
                a[i] = 1
                actions.append(a)
            if not no_op:
                actions.pop(0) # remove no op
            return actions

        # Default action set
        if action_set == "default":
            # Grab indices corresponding to buttons
            move_forward = np.where(self.action_indices == 13)
            turn_right   = np.where(self.action_indices == 14)
            turn_left    = np.where(self.action_indices == 15)
            use          = np.where(self.action_indices ==  1)
            actual_num = (np.size(move_forward) + np.size(turn_right) 
                          + np.size(turn_left) + np.size(use))
            expected_num = 4
            _check_actions(actual_num, expected_num)
            
            # Set actions array with particular button combinations
            actions = np.zeros([6,4], dtype=np.int8)
            actions[0, move_forward]               = 1
            actions[1, turn_right]                 = 1
            actions[2, turn_left]                  = 1
            actions[3, use]                        = 1
            actions[4, [move_forward, turn_right]] = 1
            actions[5, [move_forward, turn_left]]  = 1
            
        elif action_set == "basic_four":
            # Grab indices corresponding to buttons
            move_forward = np.where(self.action_indices == 13)
            turn_right   = np.where(self.action_indices == 14)
            turn_left    = np.where(self.action_indices == 15)
            use          = np.where(self.action_indices ==  1)
            actual_num = (np.size(move_forward) + np.size(turn_right) 
                         + np.size(turn_left) + np.size(use))
            expected_num = 4
            _check_actions(actual_num, expected_num)

            # Set actions array with particular button combinations
            actions = np.zeros([4,4], dtype=np.int8)
            actions[0, move_forward]               = 1
            actions[1, turn_right]                 = 1
            actions[2, turn_left]                  = 1
            actions[3, use]                        = 1

        elif action_set == "basic_three":
            # Grab indices corresponding to buttons
            move_forward = np.where(self.action_indices == 13)
            turn_right   = np.where(self.action_indices == 14)
            turn_left    = np.where(self.action_indices == 15)
            actual_num = (np.size(move_forward) + np.size(turn_right) 
                          + np.size(turn_left))
            expected_num = 3
            _check_actions(actual_num, expected_num)

            # Set actions array with particular button combinations
            actions = np.zeros([3,3], dtype=np.int8)
            actions[0, move_forward]               = 1
            actions[1, turn_right]                 = 1
            actions[2, turn_left]                  = 1

        elif action_set == "explorer":
            # Grab indices corresponding to buttons
            move_forward = np.where(self.action_indices == 13)
            turn_right   = np.where(self.action_indices == 14)
            turn_left    = np.where(self.action_indices == 15)
            actual_num = (np.size(move_forward) + np.size(turn_right) 
                          + np.size(turn_left))
            expected_num = 3
            _check_actions(actual_num, expected_num)

            # Set actions array with particular button combinations
            actions = np.zeros([3,3], dtype=np.int8)
            actions[:, move_forward]               = 1
            actions[1, turn_right]                 = 1
            actions[2, turn_left]                  = 1

        # Raise error if name not defined
        else:
            raise NameError("Name " + str(action_set) + " not found.")
        
        # Convert to list (required by DoomGame commands); kinda messy
        action_list = [] 
        for i in range(actions.shape[0]):
            action_list.append(list(actions[i,:]))
            for j in range(actions.shape[1]):
                action_list[-1][j] = np.asscalar(action_list[-1][j])
        return action_list

    def _load_agent_file(self, agent_file):
        """Grabs arguments from agent file"""
        # Open JSON file
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())

        # Convert "None" string to None type (not supported in JSON)
        recursive_dict_search(agent, "None", None)

        # TODO: implement get method to catch KeyError
        self.agent_name = agent["agent_args"]["name"]
        self.agent_type = agent["agent_args"]["type"]
        self.frame_repeat = agent["agent_args"].get(
            "frame_repeat", self.DEFAULT_AGENT_ARGS["frame_repeat"])
        self.position_timeout = agent["agent_args"].get(
            "position_timeout", self.DEFAULT_AGENT_ARGS["position_timeout"])
        self.no_op = agent["agent_args"].get(
            "no_op", self.DEFAULT_AGENT_ARGS["no_op"])
        self.net_file = agent["network_args"]["name"]
        self.net_type = agent["network_args"]["type"]
        self.alpha = agent["network_args"]["alpha"]
        self.gamma = agent["network_args"]["gamma"]
        self.phi = agent["network_args"]["phi"]
        self.channels = agent["network_args"]["channels"]
        self.reward_scale = agent["learning_args"].get(
            "reward_scale", self.DEFAULT_AGENT_ARGS["reward_scale"])
        self.shape_reward_fns = agent["learning_args"].get(
            "shape_reward_fns", self.DEFAULT_AGENT_ARGS["shape_reward_fns"])

    def _set_shape_reward(self):
        if self.shape_reward_fns is None or len(self.shape_reward_fns) == 0:
            return [lambda: None, lambda r: r]
        self.shape_reward_buffer = [0.0] * len(self.shape_reward_fns)
        init_fn_list, fn_list = [], []
        for i, fn in enumerate(self.shape_reward_fns):
            if fn[0].lower == "user1":
                def init_shape_user1():
                    self.shape_reward_buffer[i] = self.game.get_game_variable(GameVariable.USER1)
                def shape_user1(r):
                    # Credit: shaping.py @ github.com/mwydmuch/ViZDoom
                    fixed_user1 = self.game.get_game_variable(GameVariable.USER1)
                    shaping_reward = doom_fixed_to_double(fixed_user1)
                    shaping_reward = shaping_reward - self.shape_reward_buffer[i]
                    r += shaping_reward
                    self.shape_reward_buffer[i] += shaping_reward
                    return r
                init_fn_list.append(init_shape_user1)
                fn_list.append(shape_user1)
            elif fn[0].lower() == "health_loss":
                def init_shape_health_loss():
                    self.shape_reward_buffer[i] = self.game.get_game_variable(GameVariable.HEALTH)
                def shape_health_loss(r):
                    h = self.game.get_game_variable(GameVariable.HEALTH)
                    if h < self.shape_reward_buffer[i]:
                        r += fn[1] * (h - self.shape_reward_buffer[i])
                    self.shape_reward_buffer[i] = h
                    return r
                init_fn_list.append(init_shape_health_loss)
                fn_list.append(shape_health_loss)
            elif fn[0].lower() == "health_diff":
                def init_shape_health_diff():
                    self.shape_reward_buffer[i] = self.game.get_game_variable(GameVariable.HEALTH)
                def shape_health_diff(r):
                    h = self.game.get_game_variable(GameVariable.HEALTH)
                    r += fn[1] * (h - self.shape_reward_buffer[i])
                    self.shape_reward_buffer[i] = h
                    return r
                init_fn_list.append(init_shape_health_diff)
                fn_list.append(shape_health_diff)
            elif fn[0].lower() == "ammo_loss":
                def init_shape_ammo_loss():
                    self.shape_reward_buffer[i] = self.game.get_game_variable(GameVariable.AMMO1)
                def shape_ammo_loss(r):
                    ammo = self.game.get_game_variable(GameVariable.AMMO1)
                    if ammo < self.shape_reward_buffer[i]:
                        r += fn[1] * (ammo - self.shape_reward_buffer[i])
                    self.shape_reward_buffer[i] = ammo
                    return r
                init_fn_list.append(init_shape_ammo_loss)
                fn_list.append(shape_ammo_loss)
            elif fn[0].lower() == "distance":
                def init_shape_distance():
                    p_x = self.game.get_game_variable(GameVariable.POSITION_X)
                    p_y = self.game.get_game_variable(GameVariable.POSITION_Y)
                    p_z = self.game.get_game_variable(GameVariable.POSITION_Z)
                    self.shape_reward_buffer[i] = [p_x, p_y, p_z]    
                def shape_distance(r):
                    p_x = self.game.get_game_variable(GameVariable.POSITION_X)
                    p_y = self.game.get_game_variable(GameVariable.POSITION_Y)
                    p_z = self.game.get_game_variable(GameVariable.POSITION_Z)
                    p_diff = (np.asarray([p_x, p_y, p_z]) 
                                 - np.asarray(self.shape_reward_buffer[i]))
                    dist_diff = np.sqrt(np.sum(p_diff ** 2))
                    r += fn[1] * dist_diff
                    return r
                init_fn_list.append(init_shape_distance)
                fn_list.append(shape_distance)
            else:
                raise ValueError("Reward shaping type \"" + fn[0] + "\" not recognized.")

            def init_shape_reward():
                for init_fn in init_fn_list:
                    init_fn()   

            def shape_reward(r):
                for fn in fn_list:
                    r = fn(r)
                return r

        return [init_shape_reward, shape_reward]
                
    def set_train_mode(self, new_mode):
        """Sets train_mode to new value (True if training; False if testing)"""
        assert isinstance(new_mode, bool)
        self.train_mode = new_mode
        self.network.train_mode = new_mode

    def reset_state(self):
        """Resets agent state to zeros."""
        #self.state[0] = np.zeros(self.network.input_shape, dtype=np.float32)
        #self.state[1] = np.zeros([self.num_game_var], dtype=np.float32)
        self.state = self.get_zero_state()
        self.network.reset_rnn_state()

    def get_zero_state(self):
        return [np.zeros(self.network.input_shape, dtype=np.float32),
                np.zeros([self.num_game_var], dtype=np.float32)]

    def reset_history(self):
        """Resets position and action history to empty."""
        self.position_history = []
        self.action_history = []

    def _preprocess_image(self, img):
        """Converts and downsamples the input image"""
        # If channels = 1, image shape is [y, x]. Reshape to [y, x, channels]
        if img.ndim == 2:
            new_img = img[..., np.newaxis]
        
        # If channels > 1, reshape image to [y, x, channels] if not already.
        # (Required by skimage.transform)
        elif img.ndim == 3:
            if img.shape[0] == self.channels:
                new_img = np.transpose(img, [1, 2, 0])
            else:
                new_img = img
        
        # Crop image to be proportional to input resolution
        img_h = new_img.shape[0]
        img_w = new_img.shape[1]
        input_h = self.network.input_res[0]
        input_w = self.network.input_res[1]
        ratio = (img_w / img_h) / (input_w / input_h)
        if ratio > 1.0:
            # Crop image width
            crop_w = int((img_w - img_h * (input_w / input_h)) / 2)
            new_img = new_img[:, crop_w:img_w-crop_w, :]
        elif ratio < 1.0:
            # Crop image height
            crop_h = int((img_h - img_w * (input_h / input_w)) / 2)
            new_img = new_img[crop_h:img_h-crop_h, :, :]
        
        # Resize to resolution of network input and normalize to [0,1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_img = skimage.transform.resize(new_img, [input_h, input_w])
        
        # Reshape to [channels, y, x] if NCHW (i.e. on GPU)
        if self.network.data_format == "NCHW":
            new_img = np.transpose(new_img, [2, 0, 1])

        # Downsize to save memory
        new_img = new_img.astype(np.float32)

        return new_img

    def update_state(self, new_screen=None, new_game_var=None, replace=True):
        """
        Updates current state to previous phi images
        
        Args:
        - new_img: Current screen buffer of DoomGame.
        - replace (optional, default: True): Boolean. If True, add new_img to 
            state in FIFO order. If False, append to current state; assumes 
            that state has less than phi images (used for initializing state).
        """
        # Update network state if needed to reflect time (t-1)
        self.network.update_rnn_state(self.state)
        
        # Get current game variables if not specified
        if new_screen is None:
            new_screen = self.game.get_state().screen_buffer
        if new_game_var is None:
            new_game_var = self.game.get_state().game_variables
            new_game_var *= self.fixed_to_float_vector

        # Preprocess screen buffer
        new_screen = self._preprocess_image(new_screen)

        # Set data format to feed into convolutional layers
        if self.network.data_format == "NCHW":
            ax = 0
        else:
            ax = 2

        # Add new image to state. Delete least recent image if replace=True.    
        if replace:
            self.state[0] = np.delete(self.state[0], np.s_[0:self.channels], axis=ax)
            self.state[0] = np.append(self.state[0], new_screen, axis=ax)
                
        else:
            i = 0
            j = [slice(None,)] * ax
            while np.count_nonzero(self.state[0][j + [slice(i, i+1)]]) != 0:
                i += 1
            self.state[0][j + [slice(i,i+self.channels)]] = new_screen

        # Update game variables in state.
        self.state[1] = new_game_var

    def initialize_new_episode(self):
        """Starts new DoomGame episode and initialize agent state."""
        self.game.new_episode()
        self.reset_state()
        self.reset_history()
        self.position_repeat = 0
        self.initialize_shape_reward()
        for init_step in range(self.phi):
            self.update_state(replace=False)
        self.track_position()

    def check_position_timeout(self, action):
        # Check if position timeout exists
        if self.position_timeout is None:
            return action

        # Get current position
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        pos_z = self.game.get_game_variable(GameVariable.POSITION_Z)
        
        # If position unchanged, increment counter; otherwise, (re)set to zero
        try:
            if [pos_x, pos_y, pos_z] == self.position_history[-1][1:]:
                self.position_repeat += 1
            else:
                self.position_repeat = 0
        except IndexError:
            self.position_repeat = 0

        # Get random action other than previous one if timeout exceeded
        if self.position_repeat >= self.position_timeout:
            a = randint(0, self.num_actions - 1)
            action = self.actions[a]
            while action == self.action_history[-1]:
                a = randint(0, self.num_actions - 1)
                action = self.actions[a]
        
        return action

    def make_action(self, action=None, state=None, timeout=True):
        if action is None:
            # Get action from network
            if state is None: 
                state = self.state
            action = self.get_action(state)
        if timeout and self.position_timeout is not None:
            # Check position timeout
            action = self.check_position_timeout(action)
        
        # Make action, track agent, and return reward
        self.track_position()
        r = self.reward_scale * self.game.make_action(action, self.frame_repeat)
        r = self.shape_reward(r)
        self.track_action()
        return r

    def track_position(self):
        """Adds current position of agent to position history."""
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        pos_z = self.game.get_game_variable(GameVariable.POSITION_Z)
        timestamp = self.game.get_episode_time()
        pos = [pos_x, pos_y, pos_z]
        self.position_history.append([timestamp] + pos)
        return pos
    
    def track_action(self, index=False):
        """Adds current action of agent to action history."""
        timestamp = self.game.get_episode_time()
        last_action = self.game.get_last_action()
        if index:
            last_action = self.actions.index(last_action)
            self.action_history.append([timestamp, last_action])
        else:
            self.action_history.append([timestamp] + last_action)
        return last_action

    def get_score(self, var_list=None):
        """
        Returns current score of agent in episode.
        
        Args:
        - var_list (optional, default: None): List of DoomGame USER variables 
            that represent the score(s) to return. Format should be
            [GameVariable.USERXX, ...], where XX is between 1 and 60. If None,
            defaults to USER0 (which is reward visible to agent).
        
        Returns:
        - Current episode score as defined by USER variables. If argument
            contains multiple variables, then a list of scores is returned.
        """
        if var_list is None:
            return self.game.get_total_reward()
        else:
            scores = []
            for v in var_list:
                scores.append(self.game.get_game_variable(v))
            return scores 

    def get_layer_output(self, layer_output_names, state=None):
        """
        Returns values of layer(s) in network.

        Args:
        - layer_output_names: List of names of layers in network. These
            correspond to the names specified in the network file.
        - state (optional, default: None): State fed into input layer of
            network. If none, feeds current state of agent.
        
        Returns:
        - List of numpy arrays containing values of each layer specified by
            layer_output_names.
        """
        if state is None: 
            state = self.state
        return self.network.get_layer_output(state, layer_output_names)

    def get_layer_shape(self, layer_output_names):
        """
        Returns shapes of layer(s) in network.

        Args:
        - layer_output_names: List of names of layers in network. These
            correspond to the names specified in the network file.
        
        Returns:
        - List of tuples [d1, d2, ..., dn] for n-dimensional layer(s) specified 
            by layer_output_names.
        """
        return self.network.get_layer_shape(layer_output_names)

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True):
        """
        Saves current network parameters and summaries to output directory.

        Args:
        - model_name: Name of model to use in saved filenames.
        - global_step (optional, default: None): Global training step; used to
            append saved filenames ((model_name)-(global_step)).
        - save_meta (optional, default: True): Boolean. If True, save metagraph
            of network, which can be later used to restore a trained model.
        - save_summaries (optional, default: True): Boolean. If True, save 
            network summaries to display in TensorBoard. See specific network
            class for which summaries are available.
        """
        batch = None
        if save_summaries:
            batch = self._get_learning_batch()
        self.network.save_model(model_name,
                                global_step=global_step,
                                save_meta=save_meta,
                                save_summaries=save_summaries,
                                test_batch=batch)