from agent.Agent import Agent
from network.DQNetwork import DQNetwork
from memory.ReplayMemory import ReplayMemory
from memory.PrioritizedReplayMemory import PrioritizedReplayMemory
import tensorflow as tf
import numpy as np
from random import randint, random

class DQNAgent(Agent):
    
    MAIN_SCOPE = "main_network"
    TARGET_SCOPE = "target_network"

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

        # Create target network and replay memory if training
        if self.train_mode:
            # Create target network to bootstrap Q'(s', a')
            self.target_net_dir = self.net_dir + "target_net/"
            self._make_directory([self.target_net_dir])
            self.target_network = DQNetwork(phi=self.phi, 
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
            def add_transition_to_memory(s1, a, s2, isterminal, r):
                self.memory.add_transition(s1, a, s2, isterminal, r)
            
            def learn_from_memory(*args):
                # Learn from minibatch of replay memory experiences
                s1, a, target_q, w = self._get_learning_batch()
                _ = self.network.learn(s1, a, target_q)

            return add_transition_to_memory, learn_from_memory

        elif memory_type.lower() == "prioritized":
            def add_transition_to_memory(s1, a, s2, isterminal, r):
                q_ = self.network.get_q_values(s1)
                q = q_[np.arange(q_.shape[0]), a]
                target_q = self._get_target_q(s1, a, s2, isterminal, r)
                error = target_q - q
                self.memory.add_transition(s1, a, s2, isterminal, r, error)
            
            def learn_from_memory(epoch, epochs):
                # Update IS parameter β
                self.memory.update_beta(epoch, epochs)

                # Learn from minibatch of replay memory experiences
                s1, a, target_q, w = self._get_learning_batch()
                _ = self.network.learn(s1, a, target_q, weights=w)

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
        s1 = np.copy(self.state)
        
        # With probability epsilon make a random action.
        epsilon = get_exploration_rate()
        if random() <= epsilon:
            a = randint(0, self.num_actions - 1)
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
            s2 = np.zeros(self.state.shape)

        # Remember the transition that was just experienced.
        self.add_transition_to_memory(s1, a, s2, isterminal, reward)

        if self.rm_start_size < self.memory.size:
            # Update target network Q' every k steps
            if self.global_step % self.target_net_freq == 0:
                self.sess.run(self.target_update_ops)

            # Learn from minibatch of replay memory samples
            self.learn_from_memory(epoch, epoch_tot)

        self.global_step += 1        

    def _get_target_q(self, s1, a, s2, isterminal, r):
        # Update target Q for selected action using target network Q':
        # if not terminal: target_Q'(s,a) = r + gamma * max(Q'(s',_))
        # if terminal:     target_Q'(s,a) = r
        q2 = np.max(self.target_network.get_q_values(s2), axis=1)
        target_q = r + self.gamma * (1 - isterminal) * q2 
        return target_q

    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, a, s2, isterminal, r, w = self.memory.get_sample(self.batch_size)
        target_q = self._get_target_q(s1, a, s2, isterminal, r)
        return s1, a, target_q, w
    
    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True):
        batch = None
        if save_summaries:
            batch = self._get_learning_batch()
        self.network.save_model(model_name,
                                global_step=global_step,
                                save_meta=save_meta,
                                save_summaries=save_summaries,
                                test_batch=batch)
        if self.train_mode:
            self.target_network.save_model(model_name,
                                    global_step=global_step,
                                    save_meta=save_meta,
                                    save_summaries=save_summaries,
                                    test_batch=batch)