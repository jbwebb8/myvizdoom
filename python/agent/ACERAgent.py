from agent.Agent import Agent
from network.ACNetwork import ACNetwork
from memory.ReplayMemory import ReplayMemory
from memory.PrioritizedReplayMemory import PrioritizedReplayMemory
import tensorflow as tf
import numpy as np

class ACERAgent(Agent):
    
    MAIN_SCOPE = "global_network"
    TARGET_SCOPE = "target_network"

    def __init__(self, game, output_directory, agent_file=None,
                 params_file=None, train_mode=True, action_set="default", 
                 frame_repeat=4, n_step=5, **kwargs):
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
        
        # Create target network and replay memory if training
        if self.train_mode:
            # Create target network to bootstrap Q'(s', a')
            self.target_net_dir = self.net_dir + "target_net/"
            self._make_directory([self.target_net_dir])
            self.target_network = ACNetwork(phi=self.phi, 
                                            num_channels=self.channels, 
                                            num_actions=self.num_actions,
                                            learning_rate=self.alpha,
                                            network_file=self.net_file,
                                            output_directory=self.target_net_dir,
                                            session=self.sess,
                                            scope=self.TARGET_SCOPE)
            target_init_ops = self._get_target_update_ops(1.0)
            self.sess.run(target_init_ops) # copy main network initialized params
            self.target_update_ops = self._get_target_update_ops(self.target_net_rate)
            
            # Create replay memory and set specific functions
            self.memory = self._create_memory(self.rm_type)
            self.add_transition_to_memory, self.learn_from_memory \
                = self._set_memory_fns(self.rm_type)
        
        # Initialize n-step learning buffers
        self.n_step = n_step # TODO: pass this as argument
        self.s1_buffer = np.zeros([n_step] + list(self.state.shape), 
                                  dtype=np.float32)
        self.a_buffer = np.zeros(n_step, dtype=np.int32)
        self.s2_buffer = np.zeros([n_step] + list(self.state.shape),
                                  dtype=np.float32)
        self.isterminal_buffer = np.zeros(n_step, dtype=np.float32)
        self.r_buffer = np.zeros(n_step, dtype=np.float32)
        self.gamma_buffer = np.asarray([self.gamma ** k for k in range(n_step)])
        self.buffer_pos = 0
        self.episode_step = 0
    
    def _create_memory(self, memory_type):
        if memory_type.lower() == "standard":
            return ReplayMemory(capacity=self.rm_capacity, 
                                state_shape=self.state.shape,
                                input_overlap=(self.phi-1)*self.channels)
        elif memory_type.lower() == "prioritized":
            return PrioritizedReplayMemory(capacity=self.rm_capacity, 
                                           state_shape=self.state.shape,
                                           input_overlap=(self.phi-1)*self.channels)
        else:
            raise ValueError("Replay memory type \"" + memory_type + "\" not defined.")

    def _set_memory_fns(self, memory_type):
        if memory_type.lower() == "standard":
            def add_transition_to_memory(s1, a, s2, isterminal, q_sa):
                self.memory.add_transition(s1, a, s2, isterminal, q_sa)
            
            def learn_from_memory(*args):
                # Learn from minibatch of replay memory experiences
                s1, a, q_sa, w = self._get_learning_batch()
                _ = self.network.learn(s1, a, q_sa)

            return add_transition_to_memory, learn_from_memory

        elif memory_type.lower() == "prioritized":
            def add_transition_to_memory(s1, a, s2, isterminal, q_sa):
                v_s = self.target_network.get_value_output(s1)
                advantage = q_sa -v_s
                self.memory.add_transition(s1, a, s2, isterminal, q_sa, advantage)
            
            def learn_from_memory(epoch, epochs):
                # Update IS parameter β
                self.memory.update_beta(epoch, epochs)

                # Learn from minibatch of replay memory experiences
                s1, a, q_sa, w = self._get_learning_batch()
                _ = self.network.learn(s1, a, q_sa, weights=w)

            return add_transition_to_memory, learn_from_memory

    def _get_target_update_ops(self, tau):
        # Adapted from 
        # https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
        update_ops = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.MAIN_SCOPE)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.TARGET_SCOPE)
        for mv, tv in zip(main_vars, target_vars):
            update = tf.assign(tv, tau * mv.value() + (1 - tau) * tv.value())
            update_ops.append(update)
        return update_ops

    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, a, s2, isterminal, q_sa, w = self.memory.get_sample(self.batch_size)
        return s1, a, q_sa, w

    def _add_n_step_transition(self):
        # Calculate expectation of R_t-n ≈ Q(s_t-n, a_t-n):
        #      Σ(γ**i * r_i) + γ**k * V(s_t)
        t_start = self.buffer_pos - (self.n_step - 1)
        s_t = self.s2_buffer[self.buffer_pos]
        V = self.target_network.get_value_output(s_t)
        R_t_start = ( np.dot(self.gamma_buffer, self.r_buffer)
                      + (self.gamma ** self.n_step) * V )
        self.add_transition_to_memory(self.s1_buffer[t_start],
                                      self.a_buffer[t_start],
                                      self.s2_buffer[t_start], # TODO: avoid passing s2
                                      self.isterminal_buffer[t_start],
                                      R_t_start)

    def _add_terminal_transitions(self):
        running_r = 0
        for i in range(self.n_step):
            pos = self.buffer_pos - i
            running_r = self.r_buffer[pos] + (self.gamma * running_r)
            self.add_transition_to_memory(self.s1_buffer[pos],
                                          self.a_buffer[pos],
                                          self.s2_buffer[pos], # TODO: avoid passing s2
                                          self.isterminal_buffer[pos],
                                          running_r)

    def perform_learning_step(self, epoch, epoch_tot):
        # NOTE: is copying array most efficient implementation?
        s1 = np.copy(self.state)
        
        # Make action based on stochastic policy
        a = self.get_action(s1)
        
        # Receive reward from environment
        r = self.game.make_action(self.actions[a], self.frame_repeat)
        
        isterminal = self.game.is_episode_finished()
        if not isterminal:
            # Get new state
            current_screen = self.game.get_state().screen_buffer
            self.update_state(current_screen)
            s2 = np.copy(self.state)

            # Update buffers containing previous n transitions
            self.s1_buffer[self.buffer_pos] = s1
            self.a_buffer[self.buffer_pos] = a
            self.s2_buffer[self.buffer_pos] = s2
            self.isterminal_buffer[self.buffer_pos] = isterminal
            self.r_buffer[self.buffer_pos] = r
            
            if self.episode_step >= self.n_step-1: # 0-indexed
                # Store transition (t-n) in memory
                self._add_n_step_transition()
                
                # Update n-step variables
                self.gamma_buffer = np.roll(self.gamma_buffer, 1)

            self.buffer_pos = (self.buffer_pos + 1) % self.n_step
            self.episode_step += 1      
        else:
            # Terminal state set to zero
            s2 = np.zeros(self.state.shape)

            # Update buffers containing previous n transitions
            self.s1_buffer[self.buffer_pos] = s1
            self.a_buffer[self.buffer_pos] = a
            self.s2_buffer[self.buffer_pos] = s2
            self.isterminal_buffer[self.buffer_pos] = isterminal
            self.r_buffer[self.buffer_pos] = r

            # Store last n transitions (T-n,...,T-1) in memory
            self._add_terminal_transitions()

            # Reset n-step buffers and variables
            self.s1_buffer.fill(0)
            self.a_buffer.fill(0)
            self.s2_buffer.fill(0)
            self.isterminal_buffer.fill(0)
            self.r_buffer.fill(0)
            self.gamma_buffer = np.asarray([self.gamma ** k for k in range(self.n_step)])
            self.buffer_pos = 0
            self.episode_step = 0

        if self.rm_start_size < self.memory.size:
            # Update target network Q' every k steps
            if self.global_step % self.target_net_freq == 0:
                self.sess.run(self.target_update_ops)

            # Learn from minibatch of replay memory samples
            self.learn_from_memory(epoch, epoch_tot)

        self.global_step += 1    

    def get_action(self, state=None):
        if state is None: 
            state = self.state
        pi = np.squeeze(self.network.get_policy_output(state))
        return np.random.choice(np.arange(self.num_actions), p=pi)

    def make_action(self, state=None):
        if state is None: 
            state = self.state
        a = self.get_action(state)
        self.game.make_action(self.actions[a], self.frame_repeat)