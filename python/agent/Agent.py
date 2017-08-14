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
    """

    NET_JSON_DIR = "../networks/"
    MAIN_SCOPE = "main_network"
    DEFAULT_AGENT_ARGS = {"agent_name": "default",
                          "net_file":   "default",
                          "alpha":      0.00025,
                          "gamma":      0.99,
                          "phi":        1,
                          "channels":   1}

    def __init__(self, game, output_directory, agent_file=None,
                 params_file=None, train_mode=True, action_set="default", 
                 frame_repeat=4, **kwargs):
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

        # Initialize action space
        self.action_indices = np.asarray(self.game.get_available_buttons())
        self.actions = self._set_actions(action_set)
        self.num_actions = len(self.actions)
        # FIXME: how not to hard code frame_repeat?
        self.frame_repeat = frame_repeat
        
        # Load learning and network parameters
        if agent_file is not None:
            self._load_agent_file(agent_file)
        else:
            self.agent_name = kwargs.pop("agent_name", 
                                         DEFAULT_AGENT_ARGS["agent_name"])
            self.net_file   = kwargs.pop("net_name", 
                                         DEFAULT_AGENT_ARGS["net_file"])
            self.alpha      = kwargs.pop("alpha", 
                                         DEFAULT_AGENT_ARGS["alpha"])
            self.gamma      = kwargs.pop("gamma", 
                                         DEFAULT_AGENT_ARGS["gamma"])
            self.phi        = kwargs.pop("phi",
                                         DEFAULT_AGENT_ARGS["phi"])
            self.channels   = kwargs.pop("channels",
                                         DEFAULT_AGENT_ARGS["channels"])
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
                                      num_actions=self.num_actions,
                                      learning_rate=self.alpha,
                                      params_file=self.params_file,
                                      output_directory=self.main_net_dir,
                                      session=self.sess,
                                      scope=self.MAIN_SCOPE)
        self.state = np.zeros(self.network.input_shape, dtype=np.float32)

        # Create tracking lists
        self.score_history = []
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

    def _set_actions(self, action_set):
        """
        Sets available actions for agent. For dictionary of buttons and their 
        integer values, see ViZDoom/include/ViZDoomTypes.h.

        Args:
        - action_set: Name of pre-defined list of actions. Possible values are:
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
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())
        self.agent_name = agent["agent_args"]["name"]
        self.agent_type = agent["agent_args"]["type"]
        self.net_file = agent["network_args"]["name"]
        self.net_type = agent["network_args"]["type"]
        self.alpha = agent["network_args"]["alpha"]
        self.gamma = agent["network_args"]["gamma"]
        self.phi = agent["network_args"]["phi"]
        self.channels = agent["network_args"]["channels"]

    def reset_state(self):
        """Resets agent state to zeros."""
        self.state = np.zeros(self.network.input_shape, dtype=np.float32)

    def reset_history(self):
        """Resets position, action, and score history to empty."""
        self.position_history = []
        self.action_history = []
        self.score_history = []

    def _preprocess_image(self, img):
        """Converts and downsamples the input image"""
        # If channels = 1, image shape is [y, x]. Reshape to [channels, y, x]
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

    def update_state(self, new_img, replace=True):
        """
        Updates current state to previous phi images
        
        Args:
        - new_img: Current screen buffer of DoomGame.
        - replace (optional, default: True): Boolean. If True, add new_img to 
            state in FIFO order. If False, append to current state; assumes 
            that state has less than phi images (used for initializing state).
        """
        # Preprocess screen buffer
        new_state = self._preprocess_image(new_img)

        # Set data format to feed into convolutional layers
        if self.network.data_format == "NCHW":
            ax = 0
        else:
            ax = 2

        # Add new image to state. Delete least recent image if replace=True.    
        if replace:
            self.state = np.delete(self.state, np.s_[0:self.channels], axis=ax)
            self.state = np.append(self.state, new_state, axis=ax)
        else:
            i = 0
            j = [slice(None,)] * ax
            while np.count_nonzero(self.state[j + [slice(i, i+1)]]) != 0:
                i += 1
            self.state[j + [slice(i,i+self.channels)]] = new_state          

    def initialize_new_episode(self):
        """Starts new DoomGame episode and initialize agent state."""
        self.game.new_episode()
        self.reset_state()
        for init_step in range(self.phi):
            current_screen = self.game.get_state().screen_buffer
            self.update_state(current_screen, replace=False)

    def track_position(self):
        """Adds current position of agent to position history."""
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        pos_z = self.game.get_game_variable(GameVariable.POSITION_Z)
        timestamp = self.game.get_episode_time()
        self.position_history.append([timestamp, pos_x, pos_y, pos_z])
    
    def track_action(self):
        """Adds current action of agent to action history."""
        last_action = self.game.get_last_action()
        timestamp = self.game.get_episode_time()
        self.action_history.append([timestamp] + last_action)

    def update_score_history(self):
        """Adds current score of agent to score history."""
        score = self.game.get_total_reward()
        tracking_score = self.game.get_game_variable(GameVariable.USER1)
        self.score_history.append(score)

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