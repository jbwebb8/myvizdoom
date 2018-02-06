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
    def __init__(self, phi, num_channels, num_outputs, output_directory,
                 session, train_mode=True, learning_rate=None, 
                 network_file=None, params_file=None, scope=""):
        # Set basic network parameters and objects
        self.input_depth = phi * num_channels
        self.num_outputs = num_outputs
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
            self.input_shape = self.state[0].get_shape().as_list()[1:]
            self.game_var_sets = []
            for i in range(1, len(self.state)):
                self.game_var_sets.append(self.state[i].get_shape().as_list()[1])
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
               var_sum, neur_sum, grad_sum, loss_sum = builder.add_summaries()
            
            # Create objects for saving
            self.saver = tf.train.Saver(max_to_keep=None)        
            self.graph = tf.get_default_graph() 
            self.var_sum = tf.summary.merge(var_sum)
            self.neur_sum = tf.summary.merge(neur_sum)
            self.grad_sum = tf.summary.merge(grad_sum)
            self.loss_sum = tf.summary.merge(loss_sum)
            self.writer = tf.summary.FileWriter(self.log_dir, self.graph)
            self.ea = event_accumulator.EventAccumulator(self.log_dir)

        # Initialize variables or load parameters if provided
        if params_file is not None:
            self.saver.restore(self.sess, params_file)
        else:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=self.scope)
            self.sess.run(tf.variables_initializer(var_list))

    def _check_state(self, agent_state):
        """ 
        Converts agent state = [screen, game_var] to network state of form
        [screen, game_var_set_1, game_var_set_2, ...]
        """
        if agent_state is not None:
            feed_state = [] # avoids mutating agent state by reference
            
            # Check screen shape; add sample size dimension if needed
            if agent_state[0].ndim == 3:
                feed_state.append(agent_state[0].reshape([1] + list(agent_state[0].shape)))
            else:
                feed_state.append(agent_state[0])
            
            # Check game variables (if present); make column vector if needed
            i = 0
            if agent_state[1].ndim == 1:
                game_vars = agent_state[1].reshape([1] + list(agent_state[1].shape))
            else:
                game_vars = agent_state[1]
            for j in range(1, len(self.state)):
                feed_state.append(game_vars[:, i:i+self.game_var_sets[j-1]])
                i += self.game_var_sets[j-1]

            return feed_state
        
        else:
            return agent_state
    
    def _check_actions(self, actions):
        try:
            ndim = actions.ndim
        except AttributeError: # not numpy array
            actions_ = np.asarray(actions)
            ndim = actions_.ndim
            if ndim == 0: # not list or tuple
                actions_ = np.asarray([actions])
                ndim = actions_.ndim
            actions = actions_
        if actions.ndim < 2 or actions.shape[1] < 2:
            return np.column_stack([np.arange(actions.shape[0]), actions])
        else:
            return actions

    def _check_train_mode(self, feed_dict):
        """
        Adds training-dependent parameters to feed_dict if exist:
        - is_training: True if training
        - rnn_states, rnn_init_states: Uses current RNN state if testing,
          otherwise uses initial states. Ignores if no RNN in graph.
        """
        # Feed is_training if exists
        try:
            feed_dict[self.is_training] = self.train_mode
        except AttributeError:
            pass
        
        # Feed initial (training) or current (testing) RNN states if RNN defined
        try:
            if self.train_mode:
                self.reset_rnn_state(batch_size=self.train_batch_size)
                batch_size_ = self.train_batch_size
            else:
                batch_size_ = 1
            feed_dict.update({rs_: rs for rs_, rs in 
                              zip(self.rnn_states, self.rnn_current_states)})
            feed_dict[self.batch_size] = batch_size_
            
        except AttributeError:
            pass

        return feed_dict

    def reset_rnn_state(self, batch_size=1):
        """Placeholder function for RNN"""
        pass
    
    def update_rnn_state(self, s1):
        """Placeholder function for RNN"""
        pass

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
        state = self._check_state(state)
        feed_dict={s_: s for s_, s in zip(self.state, state)}
        feed_dict = self._check_train_mode(feed_dict)
        return self.sess.run(layers, feed_dict=feed_dict)
    
    def get_layer_shape(self, layer_output_names):
        layer_shapes = []
        for layer_name in layer_output_names:
            t_shape = self._get_layer(layer_name).get_shape()
            l_shape = tuple(t_shape[i].value for i in range(len(t_shape)))
            layer_shapes.append(l_shape)
        return layer_shapes