import tensorflow as tf
import json

class NetworkBuilder:

    def __init__(self, network):
        self.network = network

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
                    ph["kwargs"]["shape"] = [None, t[0], t[1], self.network.input_depth]
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"] = [None, self.network.input_depth, t[0], t[1]]
            
            # User specifies [H, W, C] or [C, H, W]
            elif len(t) == 3:
                if data_format == "NHWC":
                    ph["kwargs"]["shape"] = [None, t[0], t[1], self.network.input_depth]
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"] = [None, self.network.input_depth, t[0], t[1]]
            
            # User specifies [None, H, W, C] or [None, C, H, W]
            elif len(t) == 4:
                if data_format == "NHWC":
                    ph["kwargs"]["shape"][3] = self.network.input_depth
                elif data_format == "NCHW":
                    ph["kwargs"]["shape"][1] = self.network.input_depth
            
            else:
                raise ValueError("Unknown input format of size " + str(len(t)))
            
            return add_placeholder(ph)
        
        # Adds output layer to graph
        def add_output_layer(layer):
            layer["name"] = "Q"
            layer["kwargs"]["num_outputs"] = self.network.output_shape
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
        def add_loss_fn(loss_type, q_, a, target_q, params=None):
            if loss_type.lower() == "mean_squared_error":
                with tf.name_scope("loss"):
                    q = q_[a]
                    mse = tf.reduce_mean(tf.square(tf.subtract(target_q, q)))
                    tf.add_to_collection(tf.GraphKeys.LOSSES, mse)
                    return mse
            elif loss_type.lower() == "huber":
                with tf.name_scope("loss"):
                    delta = params[0]
                    q = q_[a]
                    error = tf.subtract(target_q, q)
                    huber_loss = tf.where(tf.abs(error) < delta, 
                                          0.5*tf.square(error),
                                          delta*(tf.abs(error) - 0.5*delta),
                                          name="huber_loss")
                    return huber_loss

            ###########################################################
            # Add new loss function support here.
            # elif loss_type.lower() == "new_loss_fn":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Loss function \"" + loss_type + "\" not yet defined.")

        # Adds optimizer to graph
        def add_optimizer(opt_type, loss):
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.network.scope)
            if opt_type.lower() == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.network.learning_rate, 
                                                      epsilon=1e-10)
                gvs = optimizer.compute_gradients(loss, var_list=var_list) # list of [grad(var), var]
                #with tf.name_scope("clip"):
                #    capped_gvs = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in gvs]
                #train_step = optimizer.apply_gradients(capped_gvs, name="train_step")
                train_step = optimizer.apply_gradients(gvs, name="train_step")
                return optimizer, train_step
            
            ###########################################################
            # Add new optimizer support here.
            # elif opt.lower() == "new_opt":
            #     return <...>
            ###########################################################

            else:
                raise ValueError("Optimizer \"" + opt_type + "\" not yet defined.")

        # Load arguments from network file
        net = json.loads(open(network_file).read())
        graph_dict = {}
        data_format = get_data_format()             

        # Add placeholders
        graph_dict["target_q"] = [tf.placeholder(tf.float32, 
                                                 shape=[None],
                                                 name="target_q"), "p"]
        graph_dict["actions"] = [tf.placeholder(tf.float32,
                                                shape=[None],
                                                name="actions"), "p"]
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
            loss_keys = net["global_features"]["loss"]
            if type(loss_keys) == list:
                loss_type = net["global_features"]["loss"][0]
                loss_params = net["global_features"]["loss"][1:]
            else:
                loss_type = net["global_features"]["loss"]
                loss_params = None
            loss_fn = add_loss_fn(loss_type=loss_type,
                                  q_=graph_dict["Q"][0],
                                  a=graph_dict["actions"][0],
                                  target_q=graph_dict["target_q"][0],
                                  params=loss_params)
            graph_dict["loss"] = [loss_fn, "o"]
        else:
            if self.network.train_mode and "loss" not in graph_dict: 
                raise ValueError("loss fn not found in network file.")

        # Add optimizer
        if "optimizer" in net["global_features"]:
            opt, ts = add_optimizer(opt_type=net["global_features"]["optimizer"],
                                    loss=graph_dict["loss"][0])
            graph_dict["optimizer"] = [opt, "s"]
            graph_dict["train_step"] = [ts, "s"]
        else:
            if self.network.train_mode:
                raise ValueError("optimizer not found in network file.") 

        return graph_dict, data_format

    def add_summaries(self):
        # Create summaries for trainable variables (weights and biases)
        var_sum = []
        with tf.name_scope("trainable_variables"):
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.network.scope):
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
            for name in self.network.graph_dict:
                if self.network.graph_dict[name][1] == "l":
                    layer = self.network.graph_dict[name][0]
                    with tf.name_scope(name):
                        num_elements = tf.cast(tf.size(layer, name="size"), tf.float64)
                        num_act = tf.cast(tf.count_nonzero(layer), tf.float64)
                        frac_act = tf.div(num_act, num_elements) # TODO: get fraction of neurons ever activated
                        neur_sum.append(tf.summary.scalar("frac_activated", frac_act))
                        neur_sum.append(tf.summary.histogram("values", layer))
        
        # Create summaries for gradients
        grad_sum = []
        with tf.name_scope("gradients"):
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                         scope=self.network.scope)
            opt = self.network.graph_dict["optimizer"][0]
            loss = self.network.graph_dict["loss"][0]
            gvs = opt.compute_gradients(loss, var_list=var_list)
            for g, v in gvs:
                with tf.name_scope(v.name[:-2]):
                    grad_sum.append(tf.summary.histogram("grads", g))
        
        return var_sum, neur_sum, grad_sum