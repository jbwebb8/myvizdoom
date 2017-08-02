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

    def __init__(self, layer_sizes, state_shape, phi, channels, num_samples=4):
        # Set toolbox parameters
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.phi = phi
        self.channels = channels

        # Initialize max data arrays
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

        # Initialize visualization tools                                      
        self.fig, self.axes = self._initialize_display()         

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

    def save_max_data(self, output_directory, layer_names=None):
        if layer_names is None:
            layer_names = ""
            for s in range(self.num_layers):
                layer_names += str(s)
        for i in range(len(layer_names)):
            layer_name = layer_names[i]
            slash = layer_names[i].find("/")
            if slash > -1:
                layer_name = layer_names[i][0:slash]                    
            np.save(results_directory + "max_values_%s-%d"
                    % (layer_name, epoch+1), 
                    max_values[i])
            np.save(results_directory + "max_states_%s-%d"
                    % (layer_name, epoch+1),
                    max_states[i])
            np.save(results_directory + "max_positions_%s-%d"
                    % (layer_name, epoch+1),
                    max_positions[i])

    def visualize_features():
        # TODO: implement visualize_features_theano.py from old_python
    
    def preprocess_state(self, state):
        if state.shape[0] == agent.phi * agent.channels:
            state = np.transpose(state, [1, 2, 0])
        imgs = np.split(state, agent.phi, axis=2)
        if agent.channels == 3:
            r = color_order.find("R")
            g = color_order.find("G")
            b = color_order.find("B")
            imgs = [imgs[i][..., [r, g, b]] for i in range(len(imgs))]
        elif agent.channels == 1:
            imgs = [np.squeeze(img) for img in imgs]
        return np.asarray(imgs)

    def _initialize_display(self, xbounds=None, ybounds=None):
        # Set up outermost components
        fig = plt.figure()
        outer = gridspec.GridSpec(2, 1)
        axes = []
        
        # Upper row:
        # One subplot per frame in phi
        # One subplot for tracking position
        inner = gridspec.GridSpecFromSubplotSpec(1, self.phi+1, subplot_spec=outer[0])
        ax = []
        for j in range(self.phi+1):
            ax_j = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax_j)
            ax_j.axis('off')
            ax.append(ax_j)
        axes.append(ax)

        # Lower row:
        # One subplot per layer visualized
        inner = gridspec.GridSpecFromSubplotSpec(1, self.num_layers+1, subplot_spec=outer[1])
        ax = []
        for j in range(self.num_layers):
            ax_j = []
            
            # Convolutional layers: grid of feature maps
            if len(self.layer_shapes[j]) == 4:
                n = int(np.ceil(np.sqrt(layer_shapes[j][1])))
                grid = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=inner[j])
                n_square = int(n*n)
                for k in range(n_square):
                    a_k = plt.Subplot(fig, grid[k])
                    fig.add_subplot(a_k)
                    a_k.axis('off')
                    ax_j.append(a_k)
            
            # Fully connected layers: single grid of neurons
            else:
                ax_j = plt.Subplot(fig, inner[j])
                fig.add_subplot(ax_j)
                ax_j.axis('off')

            ax.append(ax_j)
        axes.append(ax)
        
        # Set maze boundaries to display position
        axes[0][self.phi].set_xbound(lower=-600, upper=600)
        axes[0][self.phi].set_ybound(lower=-600, upper=600)
        axes[0][self.phi].set_aspect('equal')
 
        return fig, axes