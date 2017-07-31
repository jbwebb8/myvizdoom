from memory.ReplayMemory import ReplayMemory
import numpy as np
import math

class PrioritizedReplayMemory(ReplayMemory):

    def __init__(self, capacity, state_shape, input_overlap=0):
        # Initialize base replay memory
        ReplayMemory.__init__(self, capacity, state_shape, input_overlap)

        # Create new blank heap (see iPython notebook for details)
        heap_size = 2 ** math.ceil(math.log(capacity, 2)) + capacity
        self.start_pos = 2 ** math.ceil(math.log(capacity, 2))
        self.heap = np.zeros(heap_size, dtype=np.float32)
    
    # Overrides base ReplayMemory function with prioritization
    def add_transition(self, s1, action, s2, isterminal, reward, p):
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
        
        # Add priority of transition
        self.add_priority(p, self.pos)
        
        #Increment pointer or start over if reached end (sliding window)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    # Recursive function to update parent of node j
    def _propagate(self, child_id):
        parent_id = child_id // 2
        self.heap[parent_id] = self.heap[2 * parent_id] \
                               + self.heap[2 * parent_id + 1]

    # Add priority leaf to heap and update parent nodes
    def add_priority(self, p, i):
        # note that while heap is 1-indexed, RM is still 0-indexed
        j = self.start_pos + i 

        # Add priority of transition i to heap
        self.heap[j] = p

        # Recursively update parent nodes
        while j > 1:
            self._propagate(j)
            j = j // 2
    
    # Recursively search for node with cumulative sum range containing number
    def _retrieve(self, node, m):
        # Return value if no children (i.e. leaf)
        if 2 * node > self.heap.size - 1:
            return node

        # Move left
        if m <= self.heap[2 * node]:
            return self._retrieve(2 * node, m)

        # Move right
        else:
            m = m - self.heap[2 * node]
            return self._retrieve(2 * node + 1, m)

    # Overrides base ReplayMemory function by sampling based on priority
    def get_sample(self, sample_size):
        # Initialize matrices
        m = np.zeros(sample_size)

        # Create offset that corresponds to start value of each bin
        offset = np.zeros(sample_size)
        offset[np.arange(sample_size)] = self.heap[1] \
                                         * np.arange(sample_size) / sample_size
        
        # Draw random numbers from bin size, 
        # then add offset to create uniformly spaced distribution
        m[np.arange(sample_size)] = (self.heap[1] / sample_size) \
                                    * np.random.random(sample_size) + offset

        # Retrieve transition indices with probability proportional
        # to priority values
        t = np.zeros(sample_size, dtype=np.int32)
        for i in range(sample_size):
            t[i] = self._retrieve(1, m[i]) - self.start_pos
        t_ = start_pos + t
        p_values = self.heap[t_]
        return self.s1[t], self.a[t], self.s2[t], self.r[t], self.isterminal[t], p_values