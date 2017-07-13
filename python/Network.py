import tensorflow as tf
import json

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
    def __init__(self, phi, num_channels, output_shape, learning_rate, 
                 session, name=None, network_file=None, meta_file_path=None, 
                 params_file_path=None):
        
        self.input_depth = phi * num_channels
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.sess = session

        if network_file is not None:
            if not network_file.lower().endswith(".json"): 
                raise SyntaxError("File format not supported for network settings. " \
                                  "Please use JSON file.")
            self.name = network_file[0:-5]
            self.graph_dict = self.load_json(network_file)

        else:
            self.name = name
            self._learn, self._get_q_values, self._get_best_action \
                = self._create_network()
            self.input_shape = [self.input_depth] + self.input_res
            self.saver = tf.train.Saver()        
            self.graph = tf.get_default_graph()

    # TODO: Think how to modularize network creation. Different name for 
    # every network or broad names that allow user to specify details (like 
    # number of features, layers, etc.)

    def load_json(self, network_file):
        # Returns TensorFlow object with specified name in network file
        def _get_object(names):
            if type(names) == list:
                obs = []
                for name in names:
                    obs.append(graph_dict[name])
                return obs
            else:
                return graph_dict[names]
        
        def add_input_layer(ph):
            # Search for NCHW or NHWC format; if not found, assume NHWC
            data_format = "NHWC"
            for layer in net["layers"]:
                if "data_format" in layer["kwargs"]:
                    data_format = layer["kwargs"]["data_format"]
                    break
            if data_format = "NHWC":
                ph["kwargs"]["shape"][3] = self.input_depth
            elif data_format = "NCHW":
                ph["kwargs"]["shape"][1] = self.input_depth
            else:
                raise ValueError("Unknown data format: " + data_format)
            return add_placeholder(ph)
        
        def add_output_layer(layer):
            layer["kwargs"]["num_outputs"] = self.output_shape
            return add_layer(layer)

        # Adds placeholder to graph
        def add_placeholder(ph):
            if "shape" in ph["kwargs"]:
                for i in range(len(ph["kwargs"]["shape"])):
                    if ph["kwargs"]["shape"][i] == "None":
                        ph["kwargs"]["shape"][i] = None
            return tf.placeholder(ph["data_type"], **ph["kwargs"])
        
        # Adds layer to graph
        def add_layer(layer):
            layer_type = layer["type"].lower()
            input_layer = _get_object(layer["input"])
            if layer_type == "conv2d":
                return tf.contrib.layers.convolution2d(input_layer, **layer["kwargs"])
            elif layer_type == "flatten":
                return tf.contrib.layers.flatten(input_layer, **layer["kwargs"])
            elif layer_type == "fully_connected" or "fc":
                return tf.contrib.layers.fully_connected(input_layer, **layer["kwargs"])
            
            ###########################################################
            # Add new layer support here.
            # elif layer_type == "new_layer":
            #     return <...>
            ###########################################################
            
            else:
                raise ValueError("Layer " + layer["type"] + " not yet defined.")
        
        # Adds operation to graph
        def add_op(op):
            input_op = _get_object(op["input"])
            op_type = op["type"].lower()
            if op_type == "argmax":
                return tf.argmax(input_op, **op["kwargs"])
            elif op_type == "mean_squared_error":
                return tf.losses.mean_squared_error(*input_op, **op["kwargs"])
            
            ###########################################################
            # Add new op support here.
            # elif op_type == "new_op":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Op " + op["type"] + " not yet defined.")

        # Adds optimizer to graph
        def add_optimizer(opt):
            # Get op corresponding to loss function
            loss_name = net["global_features"]["loss"]
            loss = graph_dict[loss_name]
            
            # Create optimizer
            if opt.lower() == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                return optimizer, optimizer.minimize(loss)
            
            ###########################################################
            # Add new optimizer support here.
            # elif opt.lower() == "new_opt":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Optimizer " + opt + " not yet defined.")

        # Load arguments from network file
        net = json.loads(open(network_file).read())
        graph_dict = {}             

        # Add placeholders
        for ph in net["placeholders"]:
            if network["global_features"]["input_layer"] == ph["name"]:
                node = add_input_layer(ph) 
            else:
                node = add_placeholder(ph)
            graph_dict[ph["name"]] = node
        
        # Add layers
        for layer in net["layers"]:
            if network["global_features"]["output_layer"] == layer["name"]:
                l = add_output_layer(layer)
            else:
                l = add_layer(layer)
            graph_dict[layer["name"]] = l

        # Add ops
        for op in net["ops"]:
            node = add_op(op)
            graph_dict[op["name"]] = node

        # Add optimizer
        optimizer, train_step = add_optimizer(net["global_features"]["optimizer"])
        graph_dict["optimizer"] = optimizer
        graph_dict["train_step"] = train_step

        return graph_dict

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