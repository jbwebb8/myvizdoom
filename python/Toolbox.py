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

    def __init__(self, layer_sizes, state_shape, num_samples):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.max_values, self.max_states, self.max_positions = [], [], []
        for i in range(self.num_layers):
            self.max_values.append(np.zeros([layer_sizes[i], num_samples]))
            self.max_states.append(np.zeros([layer_sizes[i], num_samples] 
                                            + list(state_shape)))
            self.max_positions.append(np.zeros([layer_sizes[i], num_samples, 4]))          

    def _update_max_data(self, state, position, layer_values):
        if layer_values.ndim > 1:
            layer_values = layer_values.flatten()
        max_mask = (layer_values > np.amin(self.max_values, axis=1))
        max_mask = np.reshape(max_mask, [self.layer_size, 1])
        idx = np.argmin(self.max_values, axis=1)
        self.max_values[np.arange(self.layer_size), idx] = np.where(max_mask.reshape([self.layer_size,]),
                                                                    layer_values,
                                                                    self.max_values[np.arange(self.layer_size), idx])
        self.max_states[np.arange(self.layer_size), idx] = np.where(max_mask[:, :, np.newaxis, np.newaxis],
                                                                    state[np.newaxis, :, :, :],
                                                                    self.max_states[np.arange(self.layer_size), idx])
        position = np.asarray(position)
        self.max_positions[np.arange(self.layer_size), idx] = np.where(max_mask,
                                                                       position[np.newaxis, :],
                                                                       self.max_positions[np.arange(self.layer_size), idx])
    
    def update_max_data(self, state, position, layer_values):
        for i in range(self.num_layers):
            self._update_max_data(state, position, layer_values[i])


    def get_max_data(self):
        return self.max_values, self.max_states, self.max_positions

    def visualize_features():
        # TODO: implement visualize_features_theano.py from old_python
        pass