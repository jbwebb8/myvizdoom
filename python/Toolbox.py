import numpy as np

class Toolbox:
    """
    Tracks maximum activation data and assists with feature visualization
    during training and testing agents.

    Args:
    - layer_sizes: Lengths of (flattened) layers.
    - state_shape: Shape of input state.
    - num_samples: Store top k samples that best activated nodes.
    """

    def __init__(self, layer_sizes, state_shape, num_samples=4):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.max_values, self.max_states, self.max_positions = [], [], []
        # NOTE: must keep separate arrays for each layer because layer sizes
        # differ
        for i in range(self.num_layers):
            self.max_values.append(np.zeros([layer_sizes[i], num_samples],
                                            dtype=np.float64))
            self.max_states.append(np.zeros([layer_sizes[i], num_samples] 
                                            + list(state_shape), 
                                            dtype=np.float32))
            self.max_positions.append(np.zeros([layer_sizes[i], num_samples, 4],
                                               dtype=np.float32))          

    def update_max_data(self, state, position, layer_values):
        for i in range(self.num_layers):
            if layer_values[i].ndim > 1:
                layer_values[i] = layer_values[i].flatten()
            max_mask = (layer_values[i] > np.amin(self.max_values[i], axis=1))
            max_mask = np.reshape(max_mask, [self.layer_sizes[i], 1])
            idx = np.argmin(self.max_values[i], axis=1)
            self.max_values[i][np.arange(self.layer_sizes[i]), idx] \
                = np.where(max_mask.reshape([self.layer_sizes[i],]),
                           layer_values[i],
                           self.max_values[i][np.arange(self.layer_sizes[i]), idx])
            self.max_states[i][np.arange(self.layer_sizes[i]), idx] \
                = np.where(max_mask[:, :, np.newaxis, np.newaxis],
                           state[np.newaxis, :, :, :],
                           self.max_states[i][np.arange(self.layer_sizes[i]), idx])
            position = np.asarray(position)
            self.max_positions[i][np.arange(self.layer_sizes[i]), idx] \
                = np.where(max_mask,
                           position[np.newaxis, :],
                           self.max_positions[i][np.arange(self.layer_sizes[i]), idx])


    def get_max_data(self):
        return self.max_values, self.max_states, self.max_positions

    def visualize_features():
        # TODO: implement visualize_features_theano.py from old_python
        pass