from vizdoom import *
from Network import Network
from ReplayMemory import ReplayMemory
import numpy as np
import skimage.color, skimage.transform
from random import randint, random
import json
import warnings
import os, errno

class Agent:
    """
    Creates an Agent object that oversees network, memory, and learning functions.
    """

    NET_JSON_DIR = "../networks/"

    def __init__(self, game, session, output_directory, agent_file=None, 
                 train_mode=True, action_set="default", frame_repeat=4, 
                 **kwargs):
        # Initialize game
        self.game = game
        self.sess = session
        self.global_step = 0

        # Set up results directories
        if not output_directory.endswith("/"): 
            output_directory += "/"
        self.agent_dir = output_directory + "agent_data/"
        try:
            os.makedirs(self.agent_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.net_dir = output_directory + "net_data/"
        try:
            os.makedirs(self.net_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        # Initialize action space
        self.action_indices = np.asarray(self.game.get_available_buttons())
        self.actions = self._set_actions(action_set)
        self.output_size = len(self.actions)
        # FIXME: how not to hard code frame_repeat?
        self.frame_repeat = frame_repeat
        
        # Load learning and network parameters
        if agent_file is not None:
            self._load_agent_file(agent_file)
        else:
            self.agent_name = kwargs.pop("agent_name", "default")
            self.net_file = kwargs.pop("net_name", "dqn_basic")
            self.net_type = kwargs.pop("net_type", "dqn")
            self.alpha = kwargs.pop("alpha", 0.00025)
            self.gamma = kwargs.pop("gamma", 0.99)
            self.phi = kwargs.pop("phi", 1)
            self.channels = kwargs.pop("channels", 1)
            self.epsilon_start = kwargs.pop("epsilon_start", 1.0)
            self.epsilon_end = kwargs.pop("epsilon_end", 0.1)
            self.epsilon_const_rate = kwargs.pop("epsilon_const_rate", 0.1)
            self.epsilon_decay_rate = kwargs.pop("epsilon_decay_rate", 0.6)
            self.batch_size = kwargs.pop("batch_size", 64)
            self.rm_capacity = kwargs.pop("rm_capacity", 10000)
        if self.channels != self.game.get_screen_channels():
            raise ValueError("Number of image channels between agent and "
                             "game instance do not match. Please check config "
                             "and/or agent file.")
        
        # Create network components
        if not self.net_file.startswith(self.NET_JSON_DIR):
            self.net_file = self.NET_JSON_DIR + self.net_file
        if not self.net_file.endswith(".json"):
            self.net_file += ".json"
        self.network = Network(phi=self.phi, 
                               num_channels=self.channels, 
                               output_shape=self.output_size,
                               learning_rate=self.alpha,
                               session=self.sess,
                               network_file=self.net_file,
                               output_directory=self.net_dir)
        self.state = np.zeros(self.network.input_shape, dtype=np.float32)
        self.memory = ReplayMemory(capacity=self.rm_capacity, 
                                   state_shape=self.state.shape,
                                   input_overlap=(self.phi-1)*self.channels)
        
        # Create tracking lists
        self.score_history = []
        self.position_history = []
        self.action_history = []
    
    def _set_actions(self, action_set):
        # For dictionary of buttons and their integer values, see
        # ViZDoom/include/ViZDoomTypes.h
        
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
        # Grab arguments from agent file
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())
        self.agent_name = agent["agent_name"]
        self.net_file = agent["network_args"]["name"]
        self.net_type = agent["network_args"]["type"]
        self.alpha = agent["network_args"]["alpha"]
        self.gamma = agent["network_args"]["gamma"]
        self.phi = agent["network_args"]["phi"]
        self.channels = agent["network_args"]["channels"]
        self.epsilon_start = agent["learning_args"]["epsilon_start"]
        self.epsilon_end = agent["learning_args"]["epsilon_end"]
        self.epsilon_const_rate = agent["learning_args"]["epsilon_const_rate"]
        self.epsilon_decay_rate = agent["learning_args"]["epsilon_decay_rate"]
        self.batch_size = agent["learning_args"]["batch_size"]
        self.rm_capacity = agent["memory_args"]["replay_memory_size"]
    
    def reset_state(self):
        self.state = np.zeros(self.network.input_shape, dtype=np.float32)

    def reset_history(self):
        self.position_history = []
        self.action_history = []
        self.score_history = []

    # Converts and downsamples the input image
    def _preprocess_image(self, img):
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
        
        # Reshape to [channels, y, x]
        new_img = np.transpose(new_img, [2, 0, 1])

        # Downsize to save memory
        new_img = new_img.astype(np.float32)

        return new_img

    # TODO: fix for phi, num_channels > 1; need to update _preprocess_image
    # including transposing axes from [y, x, 3] to [3, y, x]
    # Updates current state to previous phi images
    def update_state(self, new_img, replace=True):
        new_state = self._preprocess_image(new_img)
        if replace:
            self.state = np.delete(self.state, np.s_[0:self.channels], axis=0)
            self.state = np.append(self.state, new_state, axis=0)
        else:
            i = 0
            while np.count_nonzero(self.state[i]) != 0:
                i += 1
            self.state[i:i+self.channels] = new_state

    def initialize_new_episode(self):
        self.game.new_episode()
        self.reset_state()
        for init_step in range(self.phi):
            current_screen = self.game.get_state().screen_buffer
            self.update_state(current_screen, replace=False)

    def perform_learning_step(self, epoch, epoch_tot):
        def get_exploration_rate(epoch, epoch_tot):
            epsilon_const_epochs = self.epsilon_const_rate * epoch_tot
            epsilon_decay_epochs = self.epsilon_decay_rate * epoch_tot
            if epoch < epsilon_const_epochs:
                return self.epsilon_start
            elif epoch < epsilon_decay_epochs:
                # Linear decay
                return (self.epsilon_start
                        + (self.epsilon_start - self.epsilon_end) / epsilon_decay_epochs 
                        * (epsilon_const_epochs - epoch))
            else:
                return self.epsilon_end
        
        # NOTE: is copying array most efficient implementation?
        s1 = np.copy(self.state)
        
        # With probability epsilon make a random action.
        epsilon = get_exploration_rate(epoch, epoch_tot)
        if random() <= epsilon:
            a = randint(0, self.output_size - 1)
        else:
            # Choose the best action according to the network.
            a = self.network.get_best_action(s1).item()
        
        # Receive reward from environment.
        reward = self.game.make_action(self.actions[a], self.frame_repeat)
        
        # Get new state if not terminal.
        isterminal = self.game.is_episode_finished()
        if not isterminal:
            current_screen = self.game.get_state().screen_buffer
            self.update_state(current_screen)
            s2 = np.copy(self.state)
        else:
            s2 = None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        # Learn from minibatch of replay memory samples.
        self.learn_from_memory()

        self.global_step += 1
    
    def learn_from_memory(self):
        # Learn from minibatch if enough memories
        if self.batch_size < self.memory.size:
            # All variables have shape [batch_size, ...]
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            q2 = np.max(self.network.get_q_values(s2), axis=1)
            # Update target Q for selected action:
            # if not terminal: target_Q(s,a) = r + gamma * max(Q(s',_))
            # if terminal:     target_Q(s,a) = r
            target_q = self.network.get_q_values(s1)
            target_q[np.arange(target_q.shape[0]), a] = r + self.gamma * (1 - isterminal) * q2
            self.network.learn(s1, target_q)
    
    def get_best_action(self, state=None):
        if state is None: 
            state = self.state
        a_best = self.network.get_best_action(state)
        return self.actions[a_best]
    
    def make_best_action(self, state=None, train_mode=True):
        if state is None: 
            state = self.state
        a_best = self.network.get_best_action(state).item()
        # Easier to use built-in feature
        if train_mode:
            self.game.make_action(self.actions[a_best])
        # Better for smooth animation if viewing
        else:
            self.game.set_action(self.actions[a_best])
            for _ in range(frame_repeat):
                self.game.advance_action()

    def track_position(self):
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        pos_z = self.game.get_game_variable(GameVariable.POSITION_Z)
        timestamp = self.game.get_episode_time()
        self.position_history.append([timestamp, pos_x, pos_y, pos_z])
    
    def track_action(self):
        last_action = self.game.get_last_action()
        timestamp = self.game.get_episode_time()
        self.action_history.append([timestamp] + last_action)

    def update_score_history(self):
        score = self.game.get_total_reward()
        tracking_score = self.game.get_game_variable(GameVariable.USER1)
        self.score_history.append(score)
    
    def save_positions(self, filename):
        np.save(self.agent_dir + filename, self.position_history)
    
    def save_actions(self, filename):
        np.save(self.agent_dir + filename, self.action_history)

    def get_layer_output(self, layer_output_names, state=None):
        if state is None: 
            state = self.state
        return self.network.get_layer_output(state, layer_output_names)

    def get_layer_shape(self, layer_output_names):
        return self.network.get_layer_shape(layer_output_names)

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True):
        test_batch=None
        if self.batch_size < self.memory.size:
            # All variables have shape [batch_size, ...]
            test_batch, _, _, _, _ = self.memory.get_sample(self.batch_size)
        self.network.save_model(model_name,
                                global_step=global_step,
                                save_meta=save_meta,
                                save_summaries=save_summaries,
                                test_batch=test_batch)