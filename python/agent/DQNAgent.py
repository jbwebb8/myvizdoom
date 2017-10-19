from agent.Agent import Agent
from network.DQNetwork import DQNetwork
from memory.ReplayMemory import ReplayMemory
from memory.PrioritizedReplayMemory import PrioritizedReplayMemory
import tensorflow as tf
import numpy as np
from random import randint, random
import json
import copy

class DQNAgent(Agent):
    """
    Creates an Agent that utilizes Q-learning via a Deep Q-Network (DQN).
    See [Mnih et al., 2015] for details of original implementation.

    Args: For general Agent args, see Agent class.
    - **kwargs (optional): May be used in place of agent file.
        - n_step (default: 1): Number of steps used in Q-learning updates.
        - epsilon_start (default: 1.0): Initial probability of taking a random
            action during training.
        - epsilon_end (default: 0.1): Final probability of taking a random
            action during training.
        - epsilon_const_rate (default: 0.1): Fraction of epochs to use 
            episilon_start initially.
        - epsilon_decay_rate (default: 0.6): Fraction of epochs after initial
            constant epochs to decay linearly to final episilon value.
        - batch_size (default: 32): Size of minibatch to sample from replay
            memory
        - target_network_update_freq (default: 4): Update target network toward
            primary network every n learning steps.
        - target_network_update_rate (default: 0.001): Fraction by which to
            update target network toward primary network.
        - memory_type (default: standard): Name of memory to use. Possible 
            values are:
            - standard: Basic replay memory that returns a random uniform
                batch of samples.
            - prioritized: Replay memory that returns samples with probability
                in proportion to their calculate temporal difference during
                learning (i.e. r + Q'(s',a') - Q(s,a)).
        - replay_memory_size (default: 10000): Max number of transitions that
            can be stored in replay memory.
        - replay_memory_start_size (default: 10000}: Number of transitions to
            store in replay memory before learning from it.
    """

    MAIN_SCOPE = "main_network"
    TARGET_SCOPE = "target_network"
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
            # Create target network to bootstrap Q'(s', a')
            self.target_net_dir = self.net_dir + "target_net/"
            self._make_directory([self.target_net_dir])
            self.target_network = DQNetwork(phi=self.phi, 
                                            num_channels=self.channels, 
                                            num_outputs=self.num_actions,
                                            learning_rate=self.alpha,
                                            network_file=self.net_file,
                                            output_directory=self.target_net_dir,
                                            session=self.sess,
                                            train_mode=self.train_mode,
                                            scope=self.TARGET_SCOPE)
            target_init_ops = self._get_target_update_ops(1.0)
            self.sess.run(target_init_ops) # copy main network initialized params
            self.target_update_ops = self._get_target_update_ops(self.target_net_rate)
            
            # Create replay memory and set specific functions
            self.memory = self._create_memory(self.rm_type)
            # NOTE: may be worth incorporating this difference as some if
            # statements rather than separate functions if other ER share
            # similar functions
            self.learn_from_memory = self._set_memory_fns(self.rm_type)
            
            # Create n-step Q-learning components
            if (not isinstance(self.n_step, int)) or (self.n_step < 1):
                raise ValueError("Agent n_step must be a natural number.")
            if self.n_step > 1:
                self.s1_buffer = ([np.zeros([self.n_step] + list(self.state[0].shape), dtype=np.float32)]
                              + [np.zeros([1], dtype=np.float32)] * self.num_game_var) * self.n_step
                self.a_buffer = np.zeros(self.n_step, dtype=np.int32)
                self.s2_buffer = ([np.zeros([self.n_step] + list(self.state[0].shape), dtype=np.float32)]
                              + [np.zeros([1], dtype=np.float32)] * self.num_game_var) * self.n_step
                self.isterminal_buffer = np.zeros(self.n_step, dtype=np.float32)
                self.r_buffer = np.zeros(self.n_step, dtype=np.float32)
                self.gamma_buffer = np.asarray([self.gamma ** k for k in range(self.n_step)])
                self.buffer_pos = 0
                self.episode_step = 0
    
    def _load_DQN_agent_file(self, agent_file):
        """Grabs DQN-specific arguments from agent file"""
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())
        self.n_step = agent["learning_args"]["n_step"]
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
            return ReplayMemory(capacity=self.rm_capacity, 
                                state_shape=self.state[0].shape,
                                num_game_var=self.num_game_var,
                                input_overlap=(self.phi-1)*self.channels)
        elif memory_type.lower() == "prioritized":
            return PrioritizedReplayMemory(capacity=self.rm_capacity, 
                                           state_shape=self.state[0].shape,
                                           num_game_var=self.num_game_var,
                                           input_overlap=(self.phi-1)*self.channels)
        else:
            raise ValueError("Replay memory type \"" + memory_type + "\" not defined.")

    def _set_memory_fns(self, memory_type):
        """Returns function to learn from memory based on memory type."""
        if memory_type.lower() == "standard":            
            def learn_from_memory(*args):
                # Learn from minibatch of replay memory experiences
                s1, a, target_q, w, idx= self._get_learning_batch()
                _ = self.network.learn(s1, a, target_q)

            return learn_from_memory

        elif memory_type.lower() == "prioritized":
            def learn_from_memory(epoch, epochs):
                # Update IS parameter β
                self.memory.update_beta(epoch, epochs)

                # Learn from minibatch of replay memory experiences
                s1, a, target_q, w, idx = self._get_learning_batch()
                q = self.network.get_q_values(s1) # before weight updates
                _ = self.network.learn(s1, a, target_q, weights=w)

                # Update priority in PER
                error = target_q - q[np.arange(q.shape[0]), np.squeeze(a)]
                self.memory.update_priority(error, idx)

            return learn_from_memory

    def _get_target_update_ops(self, tau):
        """
        Create ops in TensorFlow graph to represent updating target network
        toward primary network, given by:

        w_T = τ * w_T + (1 - τ) * w_M

        where w_T and w_M are analogous parameters in the target and main 
        networks, respectively.

        Args:
        - tau: Fraction of main network to update target network (see above).

        Returns:
        - Pointers to ops in TensorFlow graph that update target network.
        
        Credit: Adapted from 
        https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
        """
        update_ops = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.MAIN_SCOPE)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.TARGET_SCOPE)
        for mv, tv in zip(main_vars, target_vars):
            update = tf.assign(tv, tau * mv.value() + (1 - tau) * tv.value())
            update_ops.append(update)
        return update_ops
    
    def _add_n_step_transition(self, s1, a, s2, isterminal, r):
        """
        """
        
        # Update buffers containing previous n transitions
        self.s1_buffer[self.buffer_pos] = s1
        self.a_buffer[self.buffer_pos] = a
        self.s2_buffer[self.buffer_pos] = s2
        self.isterminal_buffer[self.buffer_pos] = isterminal
        self.r_buffer[self.buffer_pos] = r
        
        # Learns from expectation of R_t-n ≈ Q(s_t-n, a_t-n), given by:
        # Σ(γ**i * r_i) + γ**k * V(s_t)          
        if not isterminal:
            # After first n-steps, store transition (t-n) in memory
            if self.episode_step >= self.n_step-1: # 0-indexed
                # If s_t not terminal, store Σ(γ**i * r_i) as reward and calculate
                # γ**k * V(s_t) during learning batch with target network
                t_start = self.buffer_pos - (self.n_step - 1)
                discounted_reward = np.dot(self.gamma_buffer, self.r_buffer)
                self.memory.add_transition(self.s1_buffer[t_start],
                                           self.a_buffer[t_start],
                                           self.s2_buffer[self.buffer_pos],
                                           self.isterminal_buffer[t_start],
                                           discounted_reward)
                
                # Roll gamma buffer
                self.gamma_buffer = np.roll(self.gamma_buffer, 1)

            # Update n-step variables
            self.buffer_pos = (self.buffer_pos + 1) % self.n_step
            self.episode_step += 1 
        else:
            # If s_T terminal, store last at most n transitions (T-n,...,T-1)
            # recursively as: R_t <-- r_i + γ * R_(t+1)
            m = min(self.n_step, self.episode_step)
            running_r = 0
            for i in range(m):
                pos = self.buffer_pos - i
                running_r = self.r_buffer[pos] + (self.gamma * running_r)
                self.memory.add_transition(self.s1_buffer[pos],
                                           self.a_buffer[pos],
                                           self.s2_buffer[self.buffer_pos],
                                           self.isterminal_buffer[pos],
                                           running_r)
            
            # Reset n-step buffers and variables
            self.s1_buffer = ([np.zeros([self.n_step] + list(self.state[0].shape), dtype=np.float32)]
                              + [np.zeros([1], dtype=np.float32)] * self.num_game_var) * self.n_step
            self.a_buffer.fill(0)
            self.s2_buffer = ([np.zeros([self.n_step] + list(self.state[0].shape), dtype=np.float32)]
                              + [np.zeros([1], dtype=np.float32)] * self.num_game_var) * self.n_step
            self.isterminal_buffer.fill(0)
            self.r_buffer.fill(0)
            self.gamma_buffer = np.asarray([self.gamma ** k for k in range(self.n_step)])
            self.buffer_pos = 0
            self.episode_step = 0

    def perform_learning_step(self, epoch, epoch_tot):
        def get_exploration_rate():
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
        s1 = copy.deepcopy(self.state)
        
        # With probability epsilon make a random action; otherwise, choose
        # best action.
        epsilon = get_exploration_rate()
        if random() <= epsilon:
            a = randint(0, self.num_actions - 1)
        else:
            a = self.network.get_best_action(s1).item()
        
        # Receive reward from environment.
        r = self.make_action(action=self.actions[a])
        
        # Get new state if not terminal.
        isterminal = self.game.is_episode_finished()
        if not isterminal:
            # Get new state
            self.update_state()
            s2 = copy.deepcopy(self.state)
        else:
            # Terminal state set to zero
            s2 = [np.zeros(self.state[0].shape)] + [np.zeros([1], dtype=np.float32)] * self.num_game_var

        # Add transition to replay memory
        if self.n_step > 1:
            self._add_n_step_transition(s1, a, s2, isterminal, r)
        else:
            self.memory.add_transition(s1, a, s2, isterminal, r)

        if self.rm_start_size <= self.memory.size:
            # Update target network Q' every k steps
            if self.global_step % self.target_net_freq == 0:
                self.sess.run(self.target_update_ops)

            # Learn from minibatch of replay memory samples
            self.learn_from_memory(epoch, epoch_tot)

        self.global_step += 1        

    def _get_target_q(self, s1, a, s2, isterminal, r):
        # Update target Q for selected action using target network Q':
        # if not terminal: target_Q'(s,a) = r + gamma^n * max(Q'(s',_))
        # if terminal:     target_Q'(s,a) = r
        q2 = np.max(self.target_network.get_q_values(s2), axis=1)
        target_q = r + (self.gamma ** self.n_step) * (1 - isterminal) * q2 
        return target_q

    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, a, s2, isterminal, r, w, idx = self.memory.get_sample(self.batch_size)
        target_q = self._get_target_q(s1, a, s2, isterminal, r)
        return s1, a, target_q, w, idx
    
    def get_action(self, state=None):
        if state is None: 
            state = self.state
        a_best = self.network.get_best_action(state)[0]
        return self.actions[a_best]

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
        if save_target:
            self.target_network.save_model(model_name,
                                    global_step=global_step,
                                    save_meta=save_meta,
                                    save_summaries=save_summaries,
                                    test_batch=batch)