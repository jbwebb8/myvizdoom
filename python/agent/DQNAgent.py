from Agent import Agent

class DQNAgent(Agent):
    
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
        
        # Create target network to bootstrap Q'(s', a')
        if self.train_mode:
            self.target_net_dir = self.net_dir + "target_net/"
            self.make_directory([self.target_net_dir])
            self.target_network = Network(phi=self.phi, 
                                          num_channels=self.channels, 
                                          output_shape=self.output_size,
                                          learning_rate=self.alpha,
                                          network_file=self.net_file,
                                          output_directory=self.target_net_dir,
                                          session=self.sess,
                                          scope=self.TARGET_SCOPE)
            target_init_ops = self._get_target_update_ops(1.0)
            self.sess.run(target_init_ops) # copy main network initialized params
            self.target_update_ops = self._get_target_update_ops(self.target_net_rate)
            
            self.memory = ReplayMemory(capacity=self.rm_capacity, 
                                       state_shape=self.state.shape,
                                       input_overlap=(self.phi-1)*self.channels)

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

        # Learn from minibatch of replay memory samples and update
        # target network Q' if enough memories
        if self.rm_start_size < self.memory.size:
            self.learn_from_memory()

        self.global_step += 1        

    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
        
        # Update target Q for selected action using target network Q':
        # if not terminal: target_Q'(s,a) = r + gamma * max(Q'(s',_))
        # if terminal:     target_Q'(s,a) = r
        q2 = np.max(self.target_network.get_q_values(s2), axis=1)
        target_q = self.target_network.get_q_values(s1)
        target_q[np.arange(target_q.shape[0]), a] = r + self.gamma * (1 - isterminal) * q2

        return s1, target_q

    def learn_from_memory(self):
        # Update target network Q' every k steps
        if self.global_step % self.target_net_freq == 0:
            self.sess.run(self.target_update_ops)
        
        # Learn from minibatch of replay memory experiences
        s1, target_q = self._get_learning_batch()
        _ = self.network.learn(s1, target_q)
    
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