import numpy as np
from random import sample

class ReplayMemory:
    """
    Replay Memory
    """
    def __init__(self, 
                 capacity, 
                 state_shape, 
                 num_game_var, 
                 input_overlap=0,
                 trajectory_length=1):
        # Determine s2 shape based on overlap
        self.overlap = input_overlap
        self.chdim = np.argmin(state_shape)
        s2_shape = list(state_shape)
        s2_shape[self.chdim] = state_shape[self.chdim] - self.overlap
        
        # Initialize arrays to store transition variables
        self.s1, self.s2 = [], []
        self.s1.append(np.zeros([capacity] + list(state_shape), dtype=np.float32))
        self.s1.append(np.zeros([capacity, num_game_var], dtype=np.float32))
        self.s2.append(np.zeros([capacity] + s2_shape, dtype=np.float32))
        self.s2.append(np.zeros([capacity, num_game_var], dtype=np.float32))
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.state_shape = state_shape
        self.num_game_var = num_game_var
        self.tr_len = trajectory_length
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        # Store transition variables at current position
        self.s1[0][self.pos, ...] = s1[0]
        if not isterminal:
            # Store s2 as only the non-overlapping states along the
            # channel dimension
            self.s2[0][self.pos, ...] = s2[0][[slice(None)] * self.chdim 
                                           + [slice(self.overlap, None)]]
        self.s1[1][self.pos] = s1[1]
        self.s2[1][self.pos] = s2[1]
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        # Increment pointer or start over if reached end
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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
        