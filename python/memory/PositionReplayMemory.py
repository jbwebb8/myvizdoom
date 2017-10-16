import numpy as np
from random import sample

class PositionReplayMemory:
    """
    """
    def __init__(self, capacity, state_shape):
        # Initialize arrays to store transition variables
        self.s1 = np.zeros([capacity] + list(state_shape), dtype=np.float32)
        self.position = np.zeros([capacity, 2], dtype=np.float32)
       
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, position):
        # Store transition variables at current position
        self.s1[self.pos, ...] = s1
        self.position[self.pos] = position
        
        # Increment pointer or start over if reached end
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        # Get random minibatch of indices
        i = sample(range(0, self.size), sample_size)
        
        # Return importance sampling weights of one (stochastic distribution)
        w = np.ones([sample_size, 2])

        return self.s1[i], self.position[i], w, i