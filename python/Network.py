import tensorflow as tf

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
    """
    def __init__(self, name, phi, num_channels, output_shape, learning_rate, 
                 session, meta_file_path=None, params_file_path=None):
        self.name = name
        self.input_depth = phi * num_channels
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.sess = session

        if meta_file_path is not None:
            self.saver = tf.train.import_meta_graph(meta_file_path)
            self.saver.restore(self.sess, params_file_path)
            self.graph = tf.get_default_graph()
            # TODO: allow flexibility in choosing names of ops
            self.s1_ = self.graph.get_tensor_by_name("state:0")
            self.input_shape = self.s1_.shape[1::].as_list()
            self.input_res = self.s1_.shape[2::].as_list()
            self.a_ = self.graph.get_tensor_by_name("action:0")
            self.target_q_ = self.graph.get_tensor_by_name("target_q:0")
            self.q = self.graph.get_tensor_by_name("Q/BiasAdd:0")
            self.best_a = self.graph.get_tensor_by_name("best_a:0")
            self.loss = self.graph.get_tensor_by_name("loss/value:0")
            self.train_step = self.graph.get_operation_by_name("train_step")
            
            def _function_learn(self, s1, target_q):
                l, _ = sess.run([self.loss, self.train_step], 
                                feed_dict={self.s1_: s1, 
                                           self.target_q_: target_q})
                return l

            def _function_get_q_values(state):
                return self.sess.run(self.q, 
                                     feed_dict={self.s1_: state})

            def _function_get_best_action(state):
                if state.ndim < 4:
                    state = state.reshape([1] + list(state.shape))
                return self.sess.run(self.best_a, 
                                     feed_dict={self.s1_: state})

            self._learn = _function_learn
            self._get_q_values = _function_get_q_values
            self._get_best_action = _function_get_best_action

        elif params_file_path is not None:
            raise SyntaxError("Please include MetaGraph in which to load \
                               parameters.")
        else:
            self._learn, self._get_q_values, self._get_best_action \
                = self._create_network()
            self.input_shape = [self.input_depth] + self.input_res
            self.saver = tf.train.Saver()        
            self.graph = tf.get_default_graph()
   
    # TODO: Think how to modularize network creation. Different name for 
    # every network or broad names that allow user to specify details (like 
    # number of features, layers, etc.)
    def _create_network(self):
        if self.name.lower() == "dqn_basic":
            return self._create_dqn_basic()
        elif self.name.lower() == "dqn_lc_simple":
            return self._create_dqn_LC_simple()
        else:
            raise NameError("No network exists for " + self.name + ".")
        
    def _create_dqn_basic(self):
        self.input_res = [30, 45]

        # Create the input variables
        self.s1_ = tf.placeholder(tf.float32, [None] + [self.input_depth] + self.input_res, name="state")
        self.a_ = tf.placeholder(tf.int32, [None], name="action")
        self.target_q_ = tf.placeholder(tf.float32, [None, self.output_shape], name="target_q")
        
        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                padding="VALID",
                                                data_format="NCHW",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="CONV_1")
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                padding="VALID",
                                                data_format="NCHW",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="CONV_2")
        
        # Add FC layer
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="FC_1")
        
        # Add output layer (containing Q(s,a))
        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.output_shape, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(0.1),
                                                   scope="Q")
       
        # Define best action, loss, and optimization
        self.best_a = tf.argmax(self.q, 1, name="best_a")
        self.loss = tf.losses.mean_squared_error(self.q, self.target_q_, scope="loss")
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = optimizer.minimize(self.loss, name="train_step")

        def _function_learn(s1, target_q):
            l, _ = self.sess.run([self.loss, self.train_step], 
                                 feed_dict={self.s1_: s1, 
                                            self.target_q_: target_q})
            return l

        def _function_get_q_values(state):
            return self.sess.run(self.q, 
                                 feed_dict={self.s1_: state})

        def _function_get_best_action(state):
            if state.ndim < 4:
                state = state.reshape([1] + list(state.shape))
            return self.sess.run(self.best_a, 
                                 feed_dict={self.s1_: state})

        return _function_learn, _function_get_q_values, _function_get_best_action
    
    def _create_dqn_LC_simple(self):
        self.input_res = [60, 108]

        # Create the input variables
        self.s1_ = tf.placeholder(tf.float32, [None] + [self.input_depth] + self.input_res, name="state")
        self.a_ = tf.placeholder(tf.int32, [None], name="action")
        self.target_q_ = tf.placeholder(tf.float32, [None, self.output_shape], name="target_q")
        
        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                                padding="VALID",
                                                data_format="NCHW",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="CONV_1")
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                                                padding="VALID",
                                                data_format="NCHW",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="CONV_2")
        
        # Add FC layer
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=4608, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1),
                                                scope="FC_1")
        
        # Add output layer (containing Q(s,a))
        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.output_shape, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(0.1),
                                                   scope="Q")
        
        # Define best action, loss, and optimization
        self.best_a = tf.argmax(self.q, 1, name="best_a")
        self.loss = tf.losses.mean_squared_error(self.q, self.target_q_, scope="loss")
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = optimizer.minimize(self.loss, name="train_step")

        def _function_learn(s1, target_q):
            l, _ = self.sess.run([self.loss, self.train_step], 
                                 feed_dict={self.s1_: s1, 
                                            self.target_q_: target_q})
            return l

        def _function_get_q_values(state):
            return self.sess.run(self.q, 
                                 feed_dict={self.s1_: state})

        def _function_get_best_action(state):
            if state.ndim < 4:
                state = state.reshape([1] + list(state.shape))
            return self.sess.run(self.best_a, 
                                 feed_dict={self.s1_: state})

        return _function_learn, _function_get_q_values, _function_get_best_action

    def learn(self, s1, target_q):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._learn(s1, target_q)
    
    def get_q_values(self, state):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._get_q_values(state)
    
    def get_best_action(self, state):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._get_best_action(state)[0]
    
    def save_model(self, params_file_path, global_step=None, save_meta=True):
        self.saver.save(self.sess, params_file_path, global_step=global_step,
                        write_meta_graph=save_meta)
    
    def load_params(self, params_file_path):
        self.saver.restore(self.sess, params_file_path)
    
    def get_layer_output(self, state, layer_output_names):
        layers = []
        for layer_name in layer_output_names:
            layers.append(self.graph.get_tensor_by_name(layer_name))
        if state.ndim < 4:
            state = state.reshape([1] + list(state.shape))
        return self.sess.run(layers, feed_dict={self.s1_: state})
    
    def get_layer_shape(self, layer_output_names):
        layer_shapes = []
        for layer_name in layer_output_names:
            t_shape = self.graph.get_tensor_by_name(layer_name).get_shape()
            l_shape = tuple(t_shape[i].value for i in range(len(t_shape)))
            layer_shapes.append(l_shape)
        return layer_shapes