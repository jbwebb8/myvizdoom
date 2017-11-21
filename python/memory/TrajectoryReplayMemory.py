from memory.ReplayMemory import ReplayMemory
import numpy as np

class TrajectoryReplayMemory(ReplayMemory):
    def __init__(self, capacity, state_shape, num_game_var, input_overlap=0, 
                 trajectory_length=5):
        # Initialize base replay memory
        ReplayMemory.__init__(self, capacity, state_shape, num_game_var, input_overlap)

        # Initialize trajectory parameters
        self.tr_len = trajectory_length
        
    def get_sample(self, sample_size):
        # Get random minibatch of indices
        idx = np.random.randint(0, self.size, sample_size)
        x, y = np.meshgrid(idx, np.arange(self.tr_len))
        idx = np.transpose(x + y).flatten() # [i, i+1, ..., i+n, j, j+1, ..., j+n, k...]
        idx %= self.capacity # wrap end cases

        # TODO: find isterminal in sequences and cut short
          
        # Make list of states
        s1_sample, s2_sample = [], []

        # Get screen component
        s1_sample.append(self.s1[0][idx])
        if self.overlap > 0:
            # Stack overlapping frames from s1 to stored frames of s2 to
            # recreate full s2 state
            s2_sample.append(np.concatenate((self.s1[0][[idx] + [slice(None)] * self.chdim 
                                             + [slice(None, self.overlap)]], 
                                             self.s2[0][idx]), 
                                            axis=self.chdim+1))
        else:
            s2_sample.append(self.s2[0][idx])

        # Get game variable component
        s1_sample.append(self.s1[1][idx])
        s2_sample.append(self.s2[1][idx])

        # Get other transition parameters
        a_sample = self.a[idx]
        isterminal_sample = self.isterminal[idx]
        r_sample = self.r[idx]

        # Return importance sampling weights of one (stochastic distribution)
        w = np.ones([sample_size * self.tr_len])

        return s1_sample, a_sample, s2_sample, isterminal_sample, r_sample, w, idx
        