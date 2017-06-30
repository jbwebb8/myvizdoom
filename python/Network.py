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
            self.saver.restore(params_file_path)

            def _function_learn(self, s1, target_q):
                loss = tf.get_collection("loss")
                train_step = tf.get_collection("train_step")
                l, _ = sess.run([loss, train_step], 
                                feed_dict={s1_: s1, target_q_: target_q})
                return l

            def _function_get_q_values(state):
                q = tf.get_collection("q")
                return self.sess.run(q, feed_dict={s1_: state})

            def _function_get_best_action(state):
                if state.ndim < 4:
                    state = state.reshape([1] + list(state.shape))
                best_a = tf.get_collection("best_a")
                return self.sess.run(best_a, feed_dict={s1_: state})
            
            self._learn = _function_learn
            self._get_q_values = _function_get_q_values
            self._get_best_action = _function_get_best_action

            # TODO: set self.input_shape to input shape of loaded model
            # (required by Agent class) 

        elif params_file_path is not None:
            raise SyntaxError("Please include MetaGraph in which to load \
                               parameters.")
        else:
            self._learn, self._get_q_values, self._get_best_action \
                = self._create_network()
            self.input_shape = [self.input_depth] + self.input_res
            self.saver = tf.train.Saver()
   
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
        s1_ = tf.placeholder(tf.float32, [None] + [self.input_depth] + self.input_res, name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, self.output_shape], name="TargetQ")
        
        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        
        # Add FC layer
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))
        
        # Add output layer (containing Q(s,a))
        q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.output_shape, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
       
        # Define best action, loss, and optimization
        best_a = tf.argmax(q, 1)
        loss = tf.losses.mean_squared_error(q, target_q_)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def _function_learn(s1, target_q):
            l, _ = self.sess.run([loss, train_step], 
                                 feed_dict={s1_: s1, target_q_: target_q})
            return l

        def _function_get_q_values(state):
            return self.sess.run(q, feed_dict={s1_: state})

        def _function_get_best_action(state):
            if state.ndim < 4:
                state = state.reshape([1] + list(state.shape))
            return self.sess.run(best_a, feed_dict={s1_: state})

        return _function_learn, _function_get_q_values, _function_get_best_action
    
    def _create_dqn_LC_simple(self):
        self.input_res = [60, 108]

        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + [self.input_depth] + self.input_res, name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, self.output_shape], name="TargetQ")
        
        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=32, kernel_size=[8, 8], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        
        # Add FC layer
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=4608, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))
        
        # Add output layer (containing Q(s,a))
        q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.output_shape, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
       
        # Define best action, loss, and optimization
        best_a = tf.argmax(q, 1)
        loss = tf.losses.mean_squared_error(q, target_q_)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def _function_learn(s1, target_q):
            l, _ = self.sess.run([loss, train_step], 
                                 feed_dict={s1_: s1, target_q_: target_q})
            return l

        def _function_get_q_values(state):
            return self.sess.run(q, feed_dict={s1_: state})

        def _function_get_best_action(state):
            if state.ndim < 4:
                state = state.reshape([1] + list(state.shape))
            return self.sess.run(best_a, feed_dict={s1_: state})

        return _function_learn, _function_get_q_values, _function_get_best_action

    def learn(self, s1, target_q):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._learn(s1, target_q)
    
    def get_q_values(self, state):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._get_q_values(state)
    
    def get_best_action(self, state):
        assert self.sess != None, "TensorFlow session not assigned."
        return self._get_best_action(state)
    
    def save_model(self, params_file_path, global_step=None, save_meta=True):
        self.saver.save(self.sess, params_file_path, global_step=global_step,
                                   write_meta_graph=save_meta)
    
    def load_model(self, params_file_path):
        # TODO: create tf Saver object for saving and restoring params
        self.saver.restore(self.sess, params_file_path)