import tensorflow as tf
import json

class NetworkBuilder:

    def __init__(self, network, network_file):
        self.network = network
        self.graph_dict = {}

    # Returns TensorFlow object with specified name in network file
    def _get_object(self, names):
        if type(names) == list:
            obs = []
            for name in names:
                obs.append(self.graph_dict[name][0])
            return obs
        else:
            return self.graph_dict[names][0]
    
    # Determines data format based on file and hardware specs
    def _set_data_format(self, net_dict):
        if "data_format" in net_dict["global_features"]: 
            if net_dict["global_features"]["data_format"] == "auto":
                if tf.is_gpu_avaiable(): data_format = "NCHW"
                else:                    data_format = "NHWC" 
            else: 
                data_format = net_dict["global_features"]["data_format"]
        else:
            auto = True
            for layer in net_dict["layers"]:
                if "data_format" in layer["kwargs"]:
                    data_format = layer["kwargs"]["data_format"]
                    auto = False
                    break
            if auto:
                if tf.test.is_gpu_available(): data_format = "NCHW"
                else:                          data_format = "NHWC"
        self.data_format = data_format

    # Adds input layer to graph
    def add_input_layer(self, ph):
        ph["name"] = "state"
        t = ph["kwargs"]["shape"] # for aesthetics
        
        # User specifies [H, W]
        if len(t) == 2:
            if self.data_format == "NHWC":
                ph["kwargs"]["shape"] = [None, t[0], t[1], self.network.input_depth]
            elif self.data_format == "NCHW":
                ph["kwargs"]["shape"] = [None, self.network.input_depth, t[0], t[1]]
        
        # User specifies [H, W, C] or [C, H, W]
        elif len(t) == 3:
            if self.data_format == "NHWC":
                ph["kwargs"]["shape"] = [None, t[0], t[1], self.network.input_depth]
            elif self.data_format == "NCHW":
                ph["kwargs"]["shape"] = [None, self.network.input_depth, t[0], t[1]]
        
        # User specifies [None, H, W, C] or [None, C, H, W]
        elif len(t) == 4:
            if self.data_format == "NHWC":
                ph["kwargs"]["shape"][3] = self.network.input_depth
            elif self.data_format == "NCHW":
                ph["kwargs"]["shape"][1] = self.network.input_depth
        
        else:
            raise ValueError("Unknown input format of size " + str(len(t)))
        
        return self.add_placeholder(ph)

    # Adds placeholder to graph
    def add_placeholder(self, ph):
        if "shape" in ph["kwargs"]:
            for i in range(len(ph["kwargs"]["shape"])):
                if ph["kwargs"]["shape"][i] == "None":
                    ph["kwargs"]["shape"][i] = None
        return tf.placeholder(ph["data_type"], **ph["kwargs"])
    
    # Adds layer to graph
    def add_layer(self, layer):
        layer_type = layer["type"].lower()
        input_layer = self._get_object(layer["input"])

        # Assign custom kwargs
        if "activation_fn" in layer["kwargs"]:
            if layer["kwargs"]["activation_fn"] == "relu":
                layer["kwargs"]["activation_fn"] = tf.nn.relu
            elif layer["kwargs"]["activation_fn"] == "softmax":
                layer["kwargs"]["activation_fn"] = tf.nn.softmax
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
            elif layer["kwargs"]["biases_initializer"] == "None":
                layer["kwargs"]["biases_initializer"] = None
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
            layer["kwargs"]["data_format"] = self.data_format
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
    def add_op(self, op):
        input_op = self._get_object(op["input"])
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
    def add_loss_fn(self, loss_type, target, prediction, 
                    weights=None, params=None):
        error = tf.subtract(target, prediction, name="error")
        if weights is not None:
            error = weights * error
        if loss_type.lower() == "mean_squared_error": 
            mse = tf.reduce_mean(tf.square(error))
            tf.add_to_collection(tf.GraphKeys.LOSSES, mse)
            return mse
        elif loss_type.lower() == "huber":
            delta = params[0]
            huber_loss = tf.where(tf.abs(error) < delta, 
                                0.5*tf.square(error),
                                delta*(tf.abs(error) - 0.5*delta),
                                name="huber_loss")
            tf.add_to_collection(tf.GraphKeys.LOSSES, huber_loss)
            return huber_loss
        elif loss_type.lower() == "advantage":
            beta = tf.constant(params[0], dtype=tf.float32, name="beta")
            pi = params[1]
            pi_a = tf.gather_nd(pi, 
                                self.graph_dict["actions"][0],
                                name="pi_a")
            # Note the negative sign on adv_loss and positive (actually double
            # negative) on entropy_loss to use grad descent instead of ascent
            adv_loss = -tf.multiply(tf.log(pi_a), error, name="advantage_loss")
            entropy_loss = tf.reduce_sum(pi * tf.log(pi), name="entropy_loss")
            pi_loss_fn = tf.add(adv_loss, beta * entropy_loss,
                                name="policy_loss")
            tf.add_to_collection(tf.GraphKeys.LOSSES, pi_loss_fn)
            return pi_loss_fn

        ###########################################################
        # Add new loss function support here.
        # elif loss_type.lower() == "new_loss_fn":
        #     return <...>
        ###########################################################

        else:
            raise ValueError("Loss function \"" + loss_type + "\" not yet defined.")

    # Adds optimizer to graph
    def add_optimizer(self, opt_type, loss, var_list, *params):
        def modify_gradients(grad, mod_type, *params):
            if mod_type == "scale":
                return params[0] * grad
            elif mod_type == "clip_by_value":
                return tf.clip_by_value(grad, params[0], params[1])
            elif mod_type == "clip_by_norm":
                return tf.clip_by_norm(grad, params[0])
            else:
                raise ValueError("Gradient modification type \"" + mod_type
                                 + "not yet defined.")

        if opt_type.lower() == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.network.learning_rate, 
                                                    epsilon=1e-10)
            train_step = []
            for l in loss:
                gvs = optimizer.compute_gradients(l, var_list=var_list) # list of [grad(var), var]
                train_step.append(optimizer.apply_gradients(gvs, name="train_step"))
            return optimizer, train_step
        elif opt_type.lower() == "rmsprop_clip":
            # params = lists of [layer_name, clip_type]
            # clip_type: scale, clip_by_value, clip_by_norm
            optimizer = tf.train.RMSPropOptimizer(self.network.learning_rate, 
                                                    epsilon=1e-10)
            train_step = []
            for l in loss:
                mod_gvs = optimizer.compute_gradients(l, var_list=var_list) # list of [grad(var), var]
                for par in params:
                    layer_name = par[0]
                    clip_type = par[1]
                    if layer_name.lower() == "all":
                        mod_gvs = [[modify_gradients(g, clip_type, *params[2:]), v]
                                   for g, v in mod_gvs]
                    else:
                        mod_gvs = [[modify_gradients(g, clip_type, *params[2:]), v]
                                   if layer_name in v.name else g for g, v in mod_gvs]
                train_step.append(optimizer.apply_gradients(mod_gvs, name="train_step"))
            return optimizer, train_step
                
        ###########################################################
        # Add new optimizer support here.
        # elif opt.lower() == "new_opt":
        #     return <...>
        ###########################################################

        else:
            raise ValueError("Optimizer \"" + opt_type + "\" not yet defined.")

    def load_json(self, network_file):
        # Load arguments from network file
        net = json.loads(open(network_file).read())
        self._set_data_format(net)

        # Set specific network type
        if "type" not in net["global_features"]:
            raise SyntaxError("Please specify network type in network file.")
        net_type = net["global_features"]["type"].lower()
        if net_type == "dqn":
            builder_type = _DQN(self)
        elif net_type == "ac":
            builder_type = _AC(self)
        elif net_type == "dueling_dqn":
            builder_type = _DuelingDQN(self)
        elif net_type == "custom":
            builder_type = _Custom(self)
        else:
            raise ValueError("Unknown network type: " + net_type)

        # Add placeholders
        builder_type._add_reserved_placeholders()
        for ph in net["placeholders"]:
            if net["global_features"]["input_layer"] == ph["name"]:
                node = self.add_input_layer(ph) 
            else:
                node = self.add_placeholder(ph)
            self.graph_dict[ph["name"]] = [node, "p"]
        
        # Add layers
        for layer in net["layers"]:
            if layer["name"] in net["global_features"]["output_layer"]:
                l = builder_type._add_output_layer(layer)
            else:
                l = self.add_layer(layer)
            self.graph_dict[layer["name"]] = [l, "l"]

        # Add ops
        builder_type._add_reserved_ops()
        for op in net["ops"]:
            node = self.add_op(op)
            self.graph_dict[op["name"]] = [node, "o"]
        
        # Add loss function
        if "loss" in net["global_features"]:
            # Gather loss parameters
            loss_keys = net["global_features"]["loss"]
            if type(loss_keys) == list:
                loss_type = net["global_features"]["loss"][0]
                loss_params = net["global_features"]["loss"][1:]
            else:
                loss_type = net["global_features"]["loss"]
                loss_params = None
            
            # Add network-specific loss function
            with tf.name_scope("loss"):
                builder_type._add_loss_fn(loss_type, loss_params)
        else:
            if self.network.train_mode and "loss" not in self.graph_dict: 
                raise ValueError("loss fn not found in network file.")

        # Add optimizer
        if "optimizer" in net["global_features"]:
            # TODO: does update op keep separate rms variables for gradient
            # wrt to each loss fn?
            opt_keys = net["global_features"]["optimizer"]
            if type(opt_keys) == list:
                opt_type = net["global_features"]["optimizer"][0]
                opt_params = net["global_features"]["optimizer"][1:]
            else:
                opt_type = net["global_features"]["optimizer"]
                opt_params = None
            loss = tf.get_collection(tf.GraphKeys.LOSSES, 
                                     scope=self.network.scope)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope=self.network.scope)
            opt, ts = self.add_optimizer(opt_type=opt_type,
                                         loss=loss,
                                         var_list=var_list)
            self.graph_dict["optimizer"] = [opt, "s"]
            self.graph_dict["train_step"] = [ts, "s"]
        else:
            if self.network.train_mode:
                raise ValueError("optimizer not found in network file.") 

        return self.graph_dict, self.data_format           

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
            for name in self.graph_dict:
                if self.graph_dict[name][1] == "l":
                    layer = self.graph_dict[name][0]
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
            loss = tf.get_collection(tf.GraphKeys.LOSSES,
                                        scope=self.network.scope)
            optimizer = self.graph_dict["optimizer"][0]
            for i, l in enumerate(loss):
                gvs = optimizer.compute_gradients(l, var_list=var_list)
                for g, v in gvs:
                    with tf.name_scope(v.name[:-2]):
                        if g is not None:
                            grad_sum.append(tf.summary.histogram("grads_%d" % i, g))
            
        return var_sum, neur_sum, grad_sum

class _DQN:
    def __init__(self, network_builder):
        self.nb = network_builder

    def _add_reserved_placeholders(self):
        # <actions> must be [?, 2] rather than [None] due to tf indexing 
        # constraints (tf.gather_nd). Alternatively, could switch to 
        # indexing similar to 
        # https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
        self.nb.graph_dict["target_q"] = [tf.placeholder(tf.float32, 
                                                shape=[None],
                                                name="target_q"), "p"]
        self.nb.graph_dict["actions"] = [tf.placeholder(tf.int32,
                                                shape=[None, 2],
                                                name="actions"), "p"]

    def _add_output_layer(self, layer):
        layer["name"] = "Q"
        layer["kwargs"]["num_outputs"] = self.nb.network.num_actions
        return self.nb.add_layer(layer)                

    def _add_reserved_ops(self):
        best_action = tf.argmax(self.nb.graph_dict["Q"][0], axis=1, 
                                name="best_action")
        self.nb.graph_dict["best_action"] = [best_action, "o"]
    
    def _add_loss_fn(self, loss_type, loss_params):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")
        w = tf.placeholder(tf.float32, shape=[None], name="IS_weights")
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_type=loss_type,
                                    target=self.nb.graph_dict["target_q"][0],
                                    prediction=q_sa,
                                    weights=w,
                                    params=loss_params)
        self.nb.graph_dict["IS_weights"] = [w, "p"]
        self.nb.graph_dict["loss"] = [loss_fn, "o"]

class _DuelingDQN(_DQN):
    def __init__(self, network_builder):
        _DQN.__init__(self, network_builder)
    
    # Override DQN function
    def _add_output_layer(self, layer):
        if layer["name"].lower() == "v":
            layer["name"] = "V"
            layer["kwargs"]["num_outputs"] = 1
        elif layer["name"].lower() == "a":
            layer["name"] = "A"
            layer["kwargs"]["num_outputs"] = self.nb.network.num_actions
        return self.nb.add_layer(layer) 

    # Override DQN function
    def _add_reserved_ops(self):
        # Add Q estimation from value and advantage streams:
        # Q(s,a;θ,α,β) = V(s;θ,β) + [A(s,a;θ,α) - 1/|A| * Σ(A(s,a';θ,α))],
        # where θ, α, β are the parameters of the shared convolutional layers,
        # fully-connected stream into V, and fully-connected stream into A,
        # respectively
        V = self.nb.graph_dict["V"][0]
        A = self.nb.graph_dict["A"][0]
        with tf.name_scope("Q"):
            A_mean = tf.reduce_mean(A, axis=1)
            Q = tf.add(V, tf.subtract(A, A_mean), name="Q")
            self.nb.graph_dict["Q"] = [Q, "o"]

        # Add deterministic policy
        best_action = tf.argmax(self.nb.graph_dict["Q"][0], axis=1, 
                                name="best_action")
        self.nb.graph_dict["best_action"] = [best_action, "o"]

class _AC:
    def __init__(self, network_builder):
        self.nb = network_builder

    def _add_reserved_placeholders(self):
        # <actions> must be [?, 2] rather than [None] due to tf indexing 
        # constraints (tf.gather_nd). Alternatively, could switch to 
        # indexing similar to 
        # https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
        self.nb.graph_dict["actions"] = [tf.placeholder(tf.int32,
                                                        shape=[None, 2],
                                                        name="actions"), "p"]
        self.nb.graph_dict["q_sa"] = [tf.placeholder(tf.float32, 
                                                     shape=[None], 
                                                     name="q_sa"), "p"]
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                       shape=[None], 
                                                       name="IS_weights"), "p"]

    def _add_output_layer(self, layer):
        if layer["name"].lower() == "v":
            layer["name"] = "V"
            layer["kwargs"]["num_outputs"] = 1
        elif layer["name"].lower() == "pi":
            layer["name"] = "pi"
            layer["kwargs"]["num_outputs"] = self.nb.network.num_actions
        return self.nb.add_layer(layer)                

    def _add_reserved_ops(self):
        pass
    
    def _add_loss_fn(self, loss_type, loss_params):
        # Policy loss fn: log( π(a_t|s_t;θ') * A(a_t,s_t;θ,θ_v) )
        # where A is the advantage fn: Q(s,a) - V(s)
        # approximated by: Σ(γ**i * r_t+i) + γ**k * V(s_t+k;θ_v) - V(s_t;θ_v)
        if loss_type == "standard_ac":
            # Get inputs
            v = self.nb.graph_dict["V"][0]
            pi = self.nb.graph_dict["pi"][0]
            q_sa = self.nb.graph_dict["q_sa"][0]
            w = self.nb.graph_dict["IS_weights"][0]
            loss_params.append(pi)

            # Calculate policy loss
            pi_loss_fn = self.nb.add_loss_fn(loss_type="advantage",
                                             target=q_sa,
                                             prediction=v,
                                             weights=w,
                                             params=loss_params)
            
            # Calculate value loss
            v_loss_fn = self.nb.add_loss_fn(loss_type="mean_squared_error",
                                            target=q_sa,
                                            prediction=v,
                                            weights=w,
                                            params=loss_params)
        else:
            raise ValueError("Loss type \"" + loss_type + "not recognized.")
        
        self.nb.graph_dict["IS_weights"] = [w, "p"]
        self.nb.graph_dict["loss_pi"] = [pi_loss_fn, "o"]
        self.nb.graph_dict["loss_v"] = [v_loss_fn, "o"]

class _Custom:

    def __init__(self, network_builder):
        self.nb = network_builder

    def _add_reserved_placeholders(self):
        self.nb.graph_dict["target"] = tf.placeholder(tf.float32, 
                                                      shape=[None],
                                                      name="target")
    
    def _add_output_layer(self, layer):
        self.output_name = layer["name"]
        return self.nb.add_layer(layer)
    
    def _add_reserved_ops(self):
        pass
    
    def _add_loss_fn(self, loss_type, loss_params):
        target = self.nb.graph_dict["target"]
        prediction = self.nb.graph_dict[self.output_name]
        self.nb.add_loss_fn(loss_type, target, prediction, 
                            weights=None, params=loss_params)