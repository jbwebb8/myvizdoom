from agent.DQNAgent import DQNAgent
import numpy as np

class DRQNAgent(DQNAgent):

    #DEFAULT_DRQN_AGENT_ARGS = {"trace_length": 10}

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
        
        # Set DRQN-specific network components
        self.network.train_batch_size = self.batch_size
        if self.train_mode:
            self.target_network.train_batch_size = self.batch_size

    # Override DQN memory functions to learn from traces all at once
    # (rather than step-by-step)
    def _set_memory_fns(self, memory_type):
        """Returns function to learn from memory based on memory type."""
        if memory_type.lower() == "standard":            
            def learn_from_memory(epoch, epochs):
                # Update memory parameters
                self.memory.update_trajectory_length(epoch, epochs)

                # Get minibatch of replay memory trajectories:
                # Each has shape [batch_size * traj_len, ...], where trajectory
                # length in this case equates to trace length. Note that n-step 
                # is handled internally by the memory class, and traces are 
                # reshaped internally in the tf graph to [batch_size, trace_length, ...]
                s1, a, s2, isterminal, r, w, idx = self.memory.get_sample(self.batch_size)
                target_q = self._get_target_q(s1, a, s2, isterminal, r)
                _ = self.network.learn(s1, a, target_q, weights=w)

            return learn_from_memory

        elif memory_type.lower() == "prioritized":
            def learn_from_memory(epoch, epochs):
                # Update memory parameters
                self.memory.update_trajectory_length(epoch, epochs)
                self.memory.update_beta(epoch, epochs)

                # Get minibatch of replay memory trajectories: see note above
                s1, a, s2, isterminal, r, w, idx = self.memory.get_sample(self.batch_size)
                q = self.network.get_q_values(s1) # before weight updates
                target_q = self._get_target_q(s1, a, s2, isterminal, r) 
                _ = self.network.learn(s1, a, target_q, weights=w) 

                # Update priority in PER
                error = target_q - q[np.arange(q.shape[0]), np.squeeze(a)]
                self.memory.update_priority(error, idx)

            return learn_from_memory