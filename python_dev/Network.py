import Tensorflow as tf

class Network:
    """
    Pre-constructed neural networks

    Input:
    - name: Name of pre-constructed network (see below)
    - input_shape: Shape of input. Equivalently, this should be of shape
                   [resolution_y, resolution_x, channels]. Number of input 
                   channels into the network = (# of color channels) * (phi), 
                   where phi is the # of most recent frames in the input.
    - output_shape: Shape of output. Equivalently, this should be the number
                    of available actions.
    - session: TensorFlow session to run network functions. This can be assigned
               during construction or at a later time with tf_session().

    Output:
    - Returns a Tensorflow network

    Current networks:
    - dqn_basic: Basic DQN consisting of two convolutional layers (8 features),
                 one FC layer (size 128), and one output layer.
    """
    def _init_(self, name, input_shape, output_shape, session=None):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learn, self.get_q_values, self.get_best_action = _create_network()
        self.sess = session
    
    def tf_session(self, session):
        self.sess = session

    def _create_network(self):
        if (self.name == "dqn_basic"):
            return _create_dqn_basic()
        else:
            raise NameError("No network exists for ", self.name, ".")
        
    def _create_dqn_basic(self):
        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + [input_shape], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, output_shape], name="TargetQ")
        
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
        
        # Add output layer (containing Q(s',a))
        q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
       
        # Define best action, loss, and optimization
        best_a = tf.argmax(q, 1)
        loss = tf.losses.mean_squared_error(q, target_q_)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def _function_learn(s1, target_q):
            l, _ = session.run([loss, train_step], feed_dict={s1_: s1, target_q_: target_q})
            return l

        def _function_get_q_values(state):
            return session.run(q, feed_dict={s1_: state})

        def _function_get_best_action(state):
            return session.run(best_a, feed_dict={s1_: state})

        return _function_learn, _function_get_q_values, _function_get_best_action

    # TODO: Think how to modularize network creation. Different name for every network
    # or broad names that allow user to specify details (like features, layers, etc.)
