from helper import get_game_button_names
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

class Toolbox:
    """
    Tracks maximum activation data and assists with feature visualization
    during training and testing agents.

    Args:
    - layer_shapes: Shapes of layers.
    - state_shape: Shape of input state.
    - num_samples: Store top k samples that best activated nodes.
    """

    def __init__(self, layer_shapes, state_shape, phi, channels, actions,
                 num_samples=4, data_format="NCHW", color_format="RGB"):
        # Set toolbox parameters
        self.layer_shapes = layer_shapes
        layer_sizes = np.ones(len(layer_shapes), dtype=np.int64)
        for i in range(len(layer_shapes)):
            for j in range(len(layer_shapes[i])):
                if layer_shapes[i][j] is not None:
                    layer_sizes[i] *= layer_shapes[i][j] 
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.phi = phi
        self.channels = channels
        self.actions = actions
        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError("Unknown data format: %s" % data_format)
        self.data_format = data_format

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
        self.fig_f, self.ax_f = self._initialize_feature_display()
        self.fig_q, self.ax_q, self.idx_q, self.bars_q, self.labels_q \
            = self._initialize_q_display(self.actions)
        self.color_format = color_format
        self.prev_action = 0
        plt.ion()         

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
    
    def preprocess_state(self, state):
        if state.shape[0] == self.phi * self.channels:
            state = np.transpose(state, [1, 2, 0])
        imgs = np.split(state, self.phi, axis=2)
        if self.channels == 3:
            r = self.color_format.find("R")
            g = self.color_format.find("G")
            b = self.color_format.find("B")
            imgs = [imgs[i][..., [r, g, b]] for i in range(len(imgs))]
        elif self.channels == 1:
            imgs = [np.squeeze(img) for img in imgs]
        return np.asarray(imgs)
        
    def _initialize_feature_display(self, xbounds=[-600, 600], ybounds=[-600, 600]):
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
        inner = gridspec.GridSpecFromSubplotSpec(1, self.num_layers, subplot_spec=outer[1])
        ax = []
        for j in range(self.num_layers):
            ax_j = []
            
            # Convolutional layers: grid of feature maps
            if len(self.layer_shapes[j]) == 4:
                c_dim = self.data_format.find("C")
                n = int(np.ceil(np.sqrt(self.layer_shapes[j][c_dim])))
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
        axes[0][self.phi].set_xbound(lower=xbounds[0], upper=xbounds[1])
        axes[0][self.phi].set_ybound(lower=ybounds[0], upper=ybounds[1])
        axes[0][self.phi].set_aspect('equal')
 
        return fig, axes

    def visualize_features(self, state, position, layer_values):
        # Display state
        images = self.preprocess_state(state)
        for i in range(self.phi):
            img = self.ax_f[0][i].imshow(images[i])
        
        # Display position
        self.ax_f[0][self.phi].plot(position[1], position[2], color="black",
                                    marker='.', scalex=False, scaley=False)

        # Display layers
        for i in range(self.num_layers):
            # Convolutional layers: display activations of each feature map
            if layer_values[i].ndim == 4:
                if self.data_format == "NCHW":
                    # Layer shape = [1, features, [kernel]]
                    for j in range(layer_values[i].shape[1]):
                        mod_output = np.squeeze(layer_values[i][:, j, ...])
                        self.ax_f[1][i][j].imshow(mod_output, cmap="gray")
                elif self.data_format == "NHWC":
                    # Layer shape = [1, [kernel], features]
                    for j in range(layer_values[i].shape[3]):
                        mod_output = np.squeeze(layer_values[i][..., j])
                        self.ax_f[1][i][j].imshow(mod_output, cmap="gray")
            
            # Fully connected layers: display activations of neurons reshaped
            # into grid
            else:
                n = int(np.ceil(np.sqrt(layer_values[i].size)))
                pad = np.ones(n ** 2 - layer_values[i].size)
                mod_output = np.append(np.squeeze(layer_values[i]), pad)
                mod_output = np.reshape(mod_output, [n, -1])
                self.ax_f[1][i].imshow(mod_output, cmap="gray")
        
        # Refresh image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)
        
    def _initialize_q_display(self, actions, ybounds=[-10, 50]):
        fig, ax = plt.subplots()
        idx = np.arange(actions.size)
        # TODO: fix action label bug
        action_labels = get_game_button_names(actions)
        ax.set_title("Q Values in Real-Time")
        ax.set_xlabel("Actions")
        ax.set_xticks(idx)
        ax.set_xticklabels(action_labels)
        ax.set_ylabel("Q(s,a)")
        ax.set_ylim(ybounds)
        bars = ax.bar(idx, np.zeros(actions.size))
        labels = ax.get_xticklabels()
        return fig, ax, idx, bars, labels

    def display_q_values(self, q):
        # Clear data
        self.bars_q.remove()

        # Display Q values
        bar_width = 0.5
        self.bars_q = self.ax_q.bar(self.idx_q, q[0], bar_width, color='gray')

        # Color label of chosen axis green
        self.labels_q[self.prev_action].set_color("black")
        action = np.argmax(q[0])
        self.labels_q[action].set_color("g")
        self.prev_action = action

        # Refresh image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)
    
    def make_gif(self, images, filepath, duration=None, fps=35):
        import moviepy.editor as mpy

        def make_frame(t):
            # Grab image, accounting for rounding error
            idx = int(round(t * fps_))
            try: 
                img = images[idx]
            except IndexError: # out of bounds
                img = images[-1]

            # Convert to [H, W, C] if not already
            if img.shape[0] == self.channels:
                img = np.transpose(img, [1, 2, 0])
            
            # If color, ensure channels in RGB order;
            # if grayscale, trim to [H, W]
            if self.channels == 3:
                r = self.color_format.find("R")
                g = self.color_format.find("G")
                b = self.color_format.find("B")
                img = img[..., [r, g, b]]
            elif self.channels == 1:
                img = np.squeeze(img)

            # Rescale to [0, 255] if necessary
            if np.max(img) <= 1.0:
                img = 255.0 * img

            return img
        
        # Determine duration and/or frames per second
        if duration is not None:
            # Make clip of specified duration (not guaranteed to include
            # all images)
            duration_ = duration
            fps_ = len(images) / duration_
        else:
            # Make clip of entire sequence to match frame rate
            duration_ = len(images) / fps
            fps_ = fps
        
        # Create .gif file
        clip = mpy.VideoClip(make_frame, duration=duration_)
        if not filepath.endswith(".gif"):
            filepath += ".gif"
        clip.write_gif(filepath, fps=fps_)