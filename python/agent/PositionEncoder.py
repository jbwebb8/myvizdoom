from vizdoom import *
from agent.Agent import Agent
from network.DQNetwork import DQNetwork
from memory.ReplayMemory import ReplayMemory
from memory.PrioritizedReplayMemory import PrioritizedReplayMemory
from memory.PositionReplayMemory import PositionReplayMemory
import tensorflow as tf
import numpy as np
from random import randint, random
import json

class PositionEncoder(Agent):
    """
    """

    MAIN_SCOPE = "main_network"
    DEFAULT_DQN_AGENT_ARGS = {"n_step": 1,
                              "epsilon_start": 1.0,
                              "epsilon_end": 0.1,
                              "epsilon_const_rate": 0.1,
                              "epsilon_decay_rate": 0.6,
                              "batch_size": 32,
                              "target_network_update_freq": 4,
                              "target_network_update_rate": 0.001,
                              "replay_memory_type": "standard",
                              "replay_memory_size": 10000,
                              "replay_memory_start_size": 10000}

    def __init__(self, game, output_directory, agent_file=None,
                 params_file=None, train_mode=True, action_set="default", 
                 frame_repeat=4, **kwargs):
        # Initialize base agent class instance
        Agent.__init__(self, 
                       game, 
                       output_directory, 
                       agent_file=agent_file,
                       params_file=params_file, 
                       train_mode=train_mode, 
                       action_set=action_set, 
                       frame_repeat=frame_repeat, 
                       **kwargs)
        
        # Load DQN-specific learning and network parameters
        if agent_file is not None:
            self._load_DQN_agent_file(agent_file)
        else:
            self.n_step             = kwargs.pop("n_step",
                                                 DEFAULT_DQN_AGENT_ARGS["n_step"])
            self.epsilon_start      = kwargs.pop("epsilon_start", 
                                                 DEFAULT_DQN_AGENT_ARGS["epsilon_start"])
            self.epsilon_end        = kwargs.pop("epsilon_end", 
                                                 DEFAULT_DQN_AGENT_ARGS["epsilon_end"])
            self.epsilon_const_rate = kwargs.pop("epsilon_const_rate", 
                                                 DEFAULT_DQN_AGENT_ARGS["epsilon_const_rate"])
            self.epsilon_decay_rate = kwargs.pop("epsilon_decay_rate", 
                                                 DEFAULT_DQN_AGENT_ARGS["epsilon_decay_rate"])
            self.batch_size         = kwargs.pop("batch_size",
                                                 DEFAULT_DQN_AGENT_ARGS["batch_size"])
            self.target_net_freq    = kwargs.pop("target_network_update_freq",
                                                 DEFAULT_DQN_AGENT_ARGS["target_network_update_freq"])
            self.target_net_rate    = kwargs.pop("target_network_update_rate",
                                                 DEFAULT_DQN_AGENT_ARGS["target_network_update_rate"])
            self.rm_type            = kwargs.pop("replay_memory_type",
                                                 DEFAULT_DQN_AGENT_ARGS["replay_memory_type"])
            self.rm_capacity        = kwargs.pop("replay_memory_size",
                                                 DEFAULT_DQN_AGENT_ARGS["replay_memory_size"])
            self.rm_start_size      = kwargs.pop("replay_memory_start_size",
                                                 DEFAULT_DQN_AGENT_ARGS["replay_memory_start_size"])
                        
        # Create target network and replay memory if training
        if self.train_mode:
            # Create replay memory and set specific functions
            self.memory = self._create_memory(self.rm_type)
            # NOTE: may be worth incorporating this difference as some if
            # statements rather than separate functions if other ER share
            # similar functions
            self.learn_from_memory = self._set_memory_fns(self.rm_type)
    
    def _load_DQN_agent_file(self, agent_file):
        """Grabs DQN-specific arguments from agent file"""
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())
        self.epsilon_start = agent["learning_args"]["epsilon_start"]
        self.epsilon_end = agent["learning_args"]["epsilon_end"]
        self.epsilon_const_rate = agent["learning_args"]["epsilon_const_rate"]
        self.epsilon_decay_rate = agent["learning_args"]["epsilon_decay_rate"]
        self.batch_size = agent["learning_args"]["batch_size"]
        self.target_net_freq = agent["learning_args"]["target_network_update_freq"]
        self.target_net_rate = agent["learning_args"]["target_network_update_rate"]
        self.rm_type = agent["memory_args"]["replay_memory_type"]
        self.rm_capacity = agent["memory_args"]["replay_memory_size"]
        self.rm_start_size = agent["memory_args"]["replay_memory_start_size"]

    def _create_memory(self, memory_type):
        """
        Creates replay memory to store transitions for learning.
        
        Args:
        - memory_type (default: standard): Name of memory to use. Possible 
            values are:
            - standard: Basic replay memory that returns a random uniform
                batch of samples.
            - prioritized: Replay memory that returns samples with probability
                in proportion to their calculate temporal difference during
                learning (i.e. r + Q'(s',a') - Q(s,a)).
        
        Returns:
        - Instance of specified replay memory class.
        """
        if memory_type.lower() == "standard":
            return PositionReplayMemory(capacity=self.rm_capacity, 
                                        state_shape=self.state.shape)
        else:
            raise ValueError("Replay memory type \"" + memory_type + "\" not defined.")

    def _set_memory_fns(self, memory_type):
        """Returns function to learn from memory based on memory type."""
        if memory_type.lower() == "standard":            
            def learn_from_memory(*args):
                # Learn from minibatch of replay memory experiences
                s1, p, w, idx = self._get_learning_batch()
                _ = self.network.learn(s1, p, w)

            return learn_from_memory

    def perform_learning_step(self, epoch, epoch_tot):
        # NOTE: is copying array most efficient implementation?
        s1 = np.copy(self.state)

        # Get current position
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        # Receive reward from environment.
        a = randint(0, self.num_actions - 1)
        r = self.make_action(action=self.actions[a])
        
        # Get new state if not terminal.
        isterminal = self.game.is_episode_finished()
        if not isterminal:
            # Get new state
            current_screen = self.game.get_state().screen_buffer
            self.update_state(current_screen)        

        # Add transition to replay memory
        self.memory.add_transition(s1, [pos_x, pos_y])

        if self.rm_start_size <= self.memory.size:
            # Learn from minibatch of replay memory samples
            self.learn_from_memory(epoch, epoch_tot)

        self.global_step += 1        

    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, p, w, idx = self.memory.get_sample(self.batch_size)
        return s1, p, w, idx

    def get_action(self, state=None):
        a = randint(0, self.num_actions - 1)
        return self.actions[a]

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True, save_target=False):
        batch = None
        if save_summaries:
            batch = self._get_learning_batch()
        self.network.save_model(model_name,
                                global_step=global_step,
                                save_meta=save_meta,
                                save_summaries=save_summaries,
                                test_batch=batch)