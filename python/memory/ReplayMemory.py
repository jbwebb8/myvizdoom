import numpy as np
from random import sample

class ReplayMemory:
    """
    Replay Memory
    """
    def __init__(self, capacity, state_shape, input_overlap=0):
        # Determine s2 shape based on overlap
        self.overlap = input_overlap
        self.chdim = np.argmin(state_shape)
        s2_shape = list(state_shape)
        s2_shape[self.chdim] = state_shape[self.chdim] - self.overlap
        
        # Initialize arrays to store transition variables
        self.s1 = np.zeros([capacity] + list(state_shape), dtype=np.float32)
        self.s2 = np.zeros([capacity] + s2_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        # Store transition variables at current position
        self.s1[self.pos, ...] = s1
        self.a[self.pos] = action
        if not isterminal:
            # Store s2 as only the non-overlapping states along the
            # channel dimension
            self.s2[self.pos, ...] = s2[[slice(None)] * self.chdim 
                                        + [slice(self.overlap, None)]]
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        # Increment pointer or start over if reached end
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        # Get random minibatch of indices
        i = sample(range(0, self.size), sample_size)
        if self.overlap > 0:
            # Stack overlapping frames from s1 to stored frames of s2 to
            # recreate full s2 state
            s2 = np.concatenate((self.s1[[i] + [slice(None)] * self.chdim 
                                 + [slice(None, self.overlap)]], 
                                 self.s2[i]), 
                                 axis=self.chdim+1)
        else:
            s2 = self.s2[i]
        w = np.ones(sample_size)
        return self.s1[i], self.a[i], s2, self.isterminal[i], self.r[i], w