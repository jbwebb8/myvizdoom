import tensorflow as tf
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
    
    Reserved names and name scopes:
    - state
    - Q
    - target_q
    - loss
    - train_step
    - best_action

    Output directory branched into log directory for FileWriter and params
    directory for Saver.
    """
    def __init__(self, phi, num_channels, output_shape, output_directory,
                 train_mode=True, learning_rate=None, session=None, 
                 network_file=None):
        # Set basic network parameters and objects
        self.input_depth = phi * num_channels
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.sess = session
        self.train_mode = train_mode

        # Load JSON file
        if not network_file.lower().endswith(".json"): 
            raise SyntaxError("File format not supported for network settings. " \
                                "Please use JSON file.")
        self.name = network_file[0:-5]
        self.graph_dict, self.data_format = self.load_json(network_file)
        self.state = self.graph_dict["state"][0]
        self.input_shape = self.state.get_shape().as_list()[1:]
        if self.data_format == "NCHW":
            self.input_res = self.input_shape[1:]
        else:
            self.input_res = self.input_shape[:-1]
        self.q = self.graph_dict["Q"][0]
        self.target_q = self.graph_dict["target_q"][0]
        self.loss = self.graph_dict["loss"][0]
        self.train_step = self.graph_dict["train_step"][0]
        self.best_a = self.graph_dict["best_action"][0]

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
                
        # TODO: where to put this? Need more flexibility to add/remove
        # Create summaries for TensorBoard visualization
        with tf.name_scope("summaries"):
            # Create summaries for trainable variables (weights and biases)
            var_sum = []
            with tf.name_scope("trainable_variables"):
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    with tf.name_scope(var.name[:-2]):
                        mean = tf.reduce_mean(var)
                        var_sum.append(tf.summary.scalar("mean", mean))
                        with tf.name_scope("stddev"):
                            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                        var_sum.append(tf.summary.scalar("stddev", stddev))
                        var_sum.append(tf.summary.scalar("max", tf.reduce_max(var)))
                        var_sum.append(tf.summary.scalar("min", tf.reduce_min(var)))
                        var_sum.append(tf.summary.histogram("histogram", var))
            
            # Create summaries for neurons (% activated, values)
            neur_sum = []
            with tf.name_scope("neurons"):
                for name in self.graph_dict:
                    if self.graph_dict[name][1] == "l":
                        layer = self.graph_dict[name][0]
                        with tf.name_scope(name):
                            num_elements = tf.cast(tf.size(layer, name="size"), tf.float64)
                            num_act = tf.cast(tf.count_nonzero(layer), tf.float64)
                            frac_act = tf.div(num_act, num_elements)
                            neur_sum.append(tf.summary.scalar("frac_activated", frac_act))
                            neur_sum.append(tf.summary.histogram("values", layer))
        
        # Create objects for saving
        self.saver = tf.train.Saver()        
        self.graph = tf.get_default_graph()
        self.var_sum = tf.summary.merge(var_sum)
        self.neur_sum = tf.summary.merge(neur_sum)
        self.writer = tf.summary.FileWriter(self.log_dir, self.graph)
        
    def load_json(self, network_file):
        # Returns TensorFlow object with specified name in network file
        def _get_object(names):
            if type(names) == list:
                obs = []
                for name in names:
                    obs.append(graph_dict[name][0])
                return obs
            else:
                return graph_dict[names][0]
        
        # Determines data format based on file and hardware specs
        def get_data_format():
            if "data_format" in net["global_features"]: 
                if net["global_features"]["data_format"] == "auto":
                    if tf.is_gpu_avaiable(): data_format = "NCHW"
                    else:                    data_format = "NHWC" 
                else: 
                    data_format = net["global_features"]["data_format"]
            else:
                auto = True
                for layer in net["layers"]:
                    if "data_format" in layer["kwargs"]:
                        data_format = layer["kwargs"]["data_format"]
                        auto = False
                        break
                if auto:
                    if tf.test.is_gpu_available(): data_format = "NCHW"
                    else:                          data_format = "NHWC"
            return data_format

        # Adds input layer to graph
        def add_input_layer(ph):
            ph["name"] = "state"
            t = ph["kwargs"]["shape"] # for aesthetics
            
            # User specifies [H, W]
            if len(t) == 2:
                if data_format == "NHWC":
                    ph["kwargs"]["shape"] = [None, t[0], t[1], self.input_depth]
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"] = [None, self.input_depth, t[0], t[1]]
            
            # User specifies [H, W, C] or [C, H, W]
            elif len(t) == 3:
                if data_format == "NHWC":
                    ph["kwargs"]["shape"] = [None, t[0], t[1], self.input_depth]
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"] = [None, self.input_depth, t[0], t[1]]
            
            # User specifies [None, H, W, C] or [None, C, H, W]
            elif len(t) == 4:
                if data_format == "NHWC":
                    ph["kwargs"]["shape"][3] = self.input_depth
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"][1] = self.input_depth
            
            else:
                raise ValueError("Unknown input format of size " + str(len(t)))
            
            return add_placeholder(ph)
        
        # Adds output layer to graph
        def add_output_layer(layer):
            layer["name"] = "Q"
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

            # Assign custom kwargs
            if "activation_fn" in layer["kwargs"]:
                if layer["kwargs"]["activation_fn"] == "relu":
                    layer["kwargs"]["activation_fn"] = tf.nn.relu
                elif layer["kwargs"]["activation_fn"] == "None":
                    layer["kwargs"]["activation_fn"] = None
                else:
                    raise ValueError("Activation fn \""
                                     + layer["kwargs"]["activation_fn"] 
                                     + "\" not yet defined.")            
            if "weights_initializer" in layer["kwargs"]:
                if layer["kwargs"]["weights_initializer"] == "xavier":
                    layer["kwargs"]["weights_initializer"] = tf.contrib.layers.xavier_initializer()
                elif layer["kwargs"]["weights_initializer"][0] == "random_normal":
                    mean = float(layer["kwargs"]["weights_initializer"][1])
                    stddev = float(layer["kwargs"]["weights_initializer"][2])
                    layer["kwargs"]["weights_initializer"] = tf.random_normal_initializer(mean, stddev)
                else:
                    raise ValueError("Weights initializer \""
                                     + layer["kwargs"]["weights_initializer"] 
                                     + "\" not yet defined.")
            if "biases_initializer" in layer["kwargs"]:
                if layer["kwargs"]["biases_initializer"][0] == "constant":
                    c = float(layer["kwargs"]["biases_initializer"][1])
                    layer["kwargs"]["biases_initializer"] = tf.constant_initializer(c)
                else:
                    raise ValueError("Biases initializer \""
                                     + layer["kwargs"]["biases_initializer"] 
                                     + "\" not yet defined.")
            
            #######################################################
            # Add new kwargs support here.
            # if "custom_args" in layer["kwargs"]:
            #     ...
            #######################################################

            # Assign custom layer builds    
            if layer_type == "conv2d":
                layer["kwargs"]["data_format"] = data_format
                return tf.contrib.layers.convolution2d(input_layer, 
                                                       **layer["kwargs"])
            elif layer_type == "flatten":
                return tf.contrib.layers.flatten(input_layer, 
                                                 **layer["kwargs"])
            elif layer_type == "fully_connected" or "fc":
                return tf.contrib.layers.fully_connected(input_layer, 
                                                         **layer["kwargs"])
            
            ###########################################################
            # Add new layer support here.
            # elif layer_type == "new_layer":
            #     return <...>
            ###########################################################
            
            else:
                raise ValueError("Layer \"" + layer["type"] + "\" not yet defined.")
        
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
                raise ValueError("Op \"" + op["type"] + "\" not yet defined.")

        # Adds loss function to graph
        def add_loss_fn(loss_type, q_, target_q_):
            if loss_type.lower() == "mean_squared_error":
                return tf.losses.mean_squared_error(q_, target_q_, scope="loss")

            ###########################################################
            # Add new loss function support here.
            # elif loss_type.lower() == "new_loss_fn":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Loss function \"" + opt + "\" not yet defined.")

        # Adds optimizer to graph
        def add_optimizer(opt_type, loss):
            if opt_type.lower() == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                return optimizer, optimizer.minimize(loss)
            
            ###########################################################
            # Add new optimizer support here.
            # elif opt.lower() == "new_opt":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Optimizer \"" + opt + "\" not yet defined.")

        # Load arguments from network file
        net = json.loads(open(network_file).read())
        graph_dict = {}
        data_format = get_data_format()             

        # Add placeholders
        graph_dict["target_q"] = [tf.placeholder(tf.float32, 
                                                 shape=[None, self.output_shape],
                                                 name="target_q"), "p"]
        for ph in net["placeholders"]:
            if net["global_features"]["input_layer"] == ph["name"]:
                node = add_input_layer(ph) 
            else:
                node = add_placeholder(ph)
            graph_dict[ph["name"]] = [node, "p"]
        
        # Add layers
        for layer in net["layers"]:
            if net["global_features"]["output_layer"] == layer["name"]:
                l = add_output_layer(layer)
            else:
                l = add_layer(layer)
            graph_dict[layer["name"]] = [l, "l"]

        # Add ops
        best_action = tf.argmax(graph_dict["Q"][0], axis=1)
        graph_dict["best_action"] = [best_action, "o"]
        for op in net["ops"]:
            node = add_op(op)
            graph_dict[op["name"]] = [node, "o"]
        
        
        # Add loss function
        if "loss" in net["global_features"]:
            loss_fn = add_loss_fn(loss_type=net["global_features"]["loss"],
                                  q_=graph_dict["Q"][0],
                                  target_q_=graph_dict["target_q"][0])
            graph_dict["loss"] = [loss_fn, "o"]
        else:
            if self.train_mode: 
                raise ValueError("loss fn not found in network file.")

        # Add optimizer
        if "optimizer" in net["global_features"]:
            opt, ts = add_optimizer(opt_type=net["global_features"]["optimizer"],
                                    loss=graph_dict["loss"][0])
            graph_dict["optimizer"] = [opt, "s"]
            graph_dict["train_step"] = [ts, "s"]
        else:
            if self.train_mode:
                raise ValueError("optimizer not found in network file.") 

        return graph_dict, data_format

    def learn(self, s1_, target_q_):
        if s1_.ndim < 4:
            s1_ = s1_.reshape([1] + list(s1_.shape))
        feed_dict={self.state: s1_, self.target_q: target_q_}
        loss_, train_step_ = self.sess.run([self.loss, self.train_step],
                                           feed_dict=feed_dict)
        return loss_, train_step_
    
    def get_q_values(self, s1_):
        if s1_.ndim < 4:
            s1_ = s1_.reshape([1] + list(s1_.shape))
        feed_dict={self.state: s1_}
        return self.sess.run(self.q, feed_dict=feed_dict)
    
    def get_best_action(self, s1_):
        if s1_.ndim < 4:
            s1_ = s1_.reshape([1] + list(s1_.shape))
        feed_dict={self.state: s1_}
        return self.sess.run(self.best_a, feed_dict=feed_dict)

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True, test_batch=None):
        self.saver.save(self.sess, self.params_dir + model_name, 
                        global_step=global_step,
                        write_meta_graph=save_meta)
        if save_summaries:
            summaries = self.sess.run(self.var_sum)
            self.writer.add_summary(summaries, global_step)
            if test_batch is not None:
                feed_dict={self.state: test_batch}
                frac_activ = self.sess.run(self.neur_sum,
                                           feed_dict=feed_dict)
                self.writer.add_summary(frac_activ, global_step)


    def load_params(self, params_file_path):
        self.saver.restore(self.sess, params_file_path)
    
    def _get_layer(self, layer_name):
        in_json = False
        for l in self.graph_dict:
            if l == layer_name:
                return graph_dict[l][0]
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