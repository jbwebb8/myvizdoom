from agent.DQNAgent import DQNAgent

class DDQNAgent(DQNAgent):

    def __init__(self, game, output_directory, agent_file=None,
                 params_file=None, train_mode=True, action_set="default", 
                 frame_repeat=4, **kwargs):
        # Initialize DQNAgent class instance
        DQNAgent.__init__(self, 
                          game, 
                          output_directory, 
                          agent_file=agent_file,
                          params_file=params_file, 
                          train_mode=train_mode, 
                          action_set=action_set, 
                          frame_repeat=frame_repeat, 
                          **kwargs)

    # Override DQN update function with Double DQN update
    def _get_learning_batch(self):
        # All variables have shape [batch_size, ...]
        s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
        
        # Update target Q for selected action using target network Q':
        # if not terminal: target_Q'(s,a) = r + gamma * Q'(s', argmax{Q(s',_)})
        # if terminal:     target_Q'(s,a) = r
        a_max = np.argmax(self.network.get_q_values(s2), axis=1)
        q2 = self.target_network.get_q_values(s2)
        q2 = q2[np.arange(q2.shape[0]), a_max]
        target_q = self.target_network.get_q_values(s1)
        target_q[np.arange(target_q.shape[0]), a] = r + self.gamma * (1 - isterminal) * q2

        return s1, target_q