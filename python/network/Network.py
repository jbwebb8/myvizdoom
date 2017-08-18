from network.NetworkBuilder import NetworkBuilder
import tensorflow as tf
from tensorflow.tensorboard.backend.event_processing \
    import event_accumulator
import numpy as np
import json
import os, errno

class Network:
    """
    Builds pre-constructed neural networks.

    Args:
    - name: Name of pre-constructed network (see below).
    - phi: Number of frames stacked together to form input state.
    - num_channels: Number of channels per frame. Typically equal to 1 for
                    grayscale images and 3 for color images.
    - output_shape: Shape of output. Equivalently, this should be the number
                    of available actions.
    - session: TensorFlow session to run.

    Current networks:
    - dqn_basic: Basic DQN consisting of two convolutional layers (8 features),
                 one FC layer (size 128), and one output layer.
    - dqn_LC_simple: Modified version of network in [Lample and Chaplot, 2016]
                     with single output layer replacing LSTM and no game
                     variable secondary stream.

    Output directory branched into log directory for FileWriter and params
    directory for Saver.
    """
    def __init__(self, phi, num_channels, num_actions, output_directory,
                 session, train_mode=True, learning_rate=None, 
                 network_file=None, params_file=None, scope=""):
        # Set basic network parameters and objects
        self.input_depth = phi * num_channels
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.sess = session
        self.train_mode = train_mode
        self.scope = scope

        with tf.variable_scope(self.scope):
            # Load JSON file with NetworkBuilder
            if not network_file.lower().endswith(".json"): 
                raise SyntaxError("File format not supported for network settings. " \
                                    "Please use JSON file.")
            self.name = network_file[0:-5]
            builder = NetworkBuilder(self, network_file)
            self.graph_dict, self.data_format = builder.load_json(network_file)
            self.state = self.graph_dict["state"][0]
            self.input_shape = self.state.get_shape().as_list()[1:]
            if self.data_format == "NCHW":
                self.input_res = self.input_shape[1:]
            else:
                self.input_res = self.input_shape[:-1]
            try:
                self.is_training = self.graph_dict["is_training"][0]
            except KeyError:
                pass

            # Set output directories
            self.out_dir = output_directory
            if not self.out_dir.endswith("/"): 
                self.out_dir += "/"
            self.log_dir = self.out_dir + "log/"
            try:
                os.makedirs(self.log_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
            self.params_dir = self.out_dir + "params/"
            try:
                os.makedirs(self.params_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
                    
            # Create summaries for TensorBoard visualization
            with tf.name_scope("summaries"):
               var_sum, neur_sum, grad_sum = builder.add_summaries()
            
            # Create objects for saving
            self.saver = tf.train.Saver(max_to_keep=None)        
            self.graph = tf.get_default_graph() 
            self.var_sum = tf.summary.merge(var_sum)
            self.neur_sum = tf.summary.merge(neur_sum)
            self.grad_sum = tf.summary.merge(grad_sum)
            self.writer = tf.summary.FileWriter(self.log_dir, self.graph)
            self.ea = event_accumulator.EventAccumulator(self.log_dir)

        # Initialize variables or load parameters if provided
        if params_file is not None:
            self.saver.restore(self.sess, params_file)
        else:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=self.scope)
            self.sess.run(tf.variables_initializer(var_list))

    def _check_state(self, state):
        if state is not None and state.ndim < 4:
            return state.reshape([1] + list(state.shape))
        else:
            return state
    
    def _check_actions(self, actions):
        if actions.ndim < 2:
            return np.column_stack([np.arange(actions.shape[0]), actions])
        else:
            return actions

    def _check_train_mode(self, feed_dict):
        try:
            feed_dict[self.is_training] = self.train_mode
        except AttributeError:
            pass
        return feed_dict

    def load_params(self, params_file_path):
        self.saver.restore(self.sess, params_file_path)
    
    def _get_layer(self, layer_name):
        in_json = False
        for l in self.graph_dict:
            if l == layer_name:
                return self.graph_dict[l][0]
        return self.graph.get_tensor_by_name(layer_name)

    def get_layer_output(self, state, layer_output_names):
        layers = []
        for layer_name in layer_output_names:
            layers.append(self._get_layer(layer_name))
        if state.ndim < 4:
            state = state.reshape([1] + list(state.shape))
        return self.sess.run(layers, feed_dict={self.state: state})
    
    def get_layer_shape(self, layer_output_names):
        layer_shapes = []
        for layer_name in layer_output_names:
            t_shape = self._get_layer(layer_name).get_shape()
            l_shape = tuple(t_shape[i].value for i in range(len(t_shape)))
            layer_shapes.append(l_shape)
        return layer_shapes