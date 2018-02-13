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
                 trajectory_length=1,
                 n_step=1,
                 aux_var_shapes=[]):
        # Store memory parameters
        self.state_shape = state_shape
        self.num_game_var = num_game_var
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.lap = 0
        self.n_step = n_step
        
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
        self.aux_vars = [] # auxiliary variables to additionally be stored
        self.add_auxiliary_variables(aux_var_shapes)

        # Get trajectory parameter(s) of either form:
        # A) [start_length, stop_length, const_fraction, decay_fraction]
        if isinstance(trajectory_length, list):
            # Perform error checks
            if len(trajectory_length) != 4:
                raise ValueError("Replay memory trajectory length must be "
                                 + "integer or list [start, stop, const_frac, "
                                 + "decay_frac].")
            elif trajectory_length[2] + trajectory_length[3] > 1.0:
                raise ValueError("Trajectory length constant fraction + "
                                 + "decay fraction must be less than or equal "
                                 + "to 1.")
            elif trajectory_length[0] < 1 or trajectory_length[1] < 1:
                raise ValueError("Trajectory length must be >= 1.")

            # Set parameters
            self.tr_len_start = trajectory_length[0]
            self.tr_len_stop = trajectory_length[1]
            self.tr_len_const = trajectory_length[2]
            self.tr_len_decay = trajectory_length[3]
            self.tr_len = self.tr_len_start
        # B) constant_trajectory_length
        else:
            # Perform error checks
            if trajectory_length < 1:
                raise ValueError("Trajectory length must be >= 1.")

            # Set parameters
            self.tr_len_start = trajectory_length
            self.tr_len_stop = trajectory_length
            self.tr_len_const = 1.0
            self.tr_len_decay = 0.0
            self.tr_len = self.tr_len_start 

    def add_auxiliary_variables(self, aux_var_shapes):
        for s in aux_var_shapes:
            if not isinstance(s, list):
                s = [s]
            self.aux_vars.append(np.zeros([self.capacity] + s, dtype=np.float32))

    def update_trajectory_length(self, epoch, total_epochs):
        frac_train = epoch / total_epochs
        if frac_train < self.tr_len_const:
            self.tr_len = self.tr_len_start
        elif frac_train < self.tr_len_const + self.tr_len_decay:
            self.tr_len = ((self.tr_len_stop - self.tr_len_start)
                           * (frac_train - self.tr_len_const) / self.tr_len_decay
                           + self.tr_len_start)
            self.tr_len = round(self.tr_len)
        else:
            self.tr_len = self.tr_len_stop
        
        # Cast as int to avoid indexing errors
        if not isinstance(self.tr_len, int):
            self.tr_len = int(self.tr_len)

    def add_transition(self, s1, action, s2, isterminal, reward, *args):
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

        # Store auxiliary variables if specified
        if len(args) != len(self.aux_vars):
            raise SyntaxError("Number of auxiliary variables does not match"
                              + "number of arguments: %d vars, %d args"
                              % (len(self.aux_vars), len(args)))
        for i, aux_var in enumerate(self.aux_vars):
            aux_var[self.pos] = args[i]

        # Increment pointer or start over if reached end
        self.pos = (self.pos + 1) % self.capacity
        self.lap += (self.pos == 0)
        self.size = min(self.size + 1, self.capacity)

    def _get_valid_idx(self, idx, valid_idx, side='right'):
        # Check intent since 'right' assumed below
        if side not in ['left', 'right']:
            raise ValueError("Undefined value \"" + side + "\" for arg 'side'.")
        
        # Round to nearest right-sided valid idx doing left-sided search
        valid_idx = np.sort(valid_idx) # sort if not already sorted
        new_idx_ = np.searchsorted(valid_idx, idx, side='left') # right-sided round
        new_idx_ %= len(valid_idx) # wrap end case
        
        # Round to nearest left-sided valid idx if specified
        if side == 'left':
            new_idx = valid_idx[new_idx_]
            new_idx_ = new_idx_ - (new_idx != idx) # left shift if not equal
        
        return valid_idx[new_idx_]

    def _get_valid_idx_trajectories(self, idx, idx_start=None, idx_end=None):
        # If shifted to valid start values, move forward in trajectory
        if idx_start is not None:
            idx = self._get_valid_idx(idx, idx_start, side='left')
            x, y = np.meshgrid(idx, np.arange(self.tr_len) * self.n_step) # non-overlapping n-step sequences
            idx = np.transpose(x + y).flatten() # [i, i+1, ..., i+n, j, j+1, ..., j+n, k...]
            idx %= self.capacity # wrap end cases
        
        # Otherwise, move backward in trajectory
        else:
            if idx_end is not None:
                idx = self._get_valid_idx(idx, idx_end, side='right')
            x, y = np.meshgrid(idx, np.arange(self.tr_len) * self.n_step) # non-overlapping n-step sequences
            idx = np.transpose(x - y).flatten() # [i-n, i-n+1, ..., i, j-n, j-n+1, ..., j, ...]
        
        return idx

    def get_sample(self, sample_size, idx_start=None, idx_end=None):
        # Get random minibatch of indices
        idx = np.random.randint(0, self.size, sample_size)

        # Extend idx to cover (valid) trajectories
        idx = self._get_valid_idx_trajectories(idx, 
                                               idx_start=idx_start, 
                                               idx_end=idx_end)

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
        aux_var_sample = []
        for aux_var in self.aux_vars:
            aux_var_sample.append(aux_var[idx])

        # Return importance sampling weights of one (stochastic distribution)
        w = np.ones([sample_size * self.tr_len])

        return (s1_sample, a_sample, s2_sample, isterminal_sample, r_sample,
                w, idx, *aux_var_sample)
        