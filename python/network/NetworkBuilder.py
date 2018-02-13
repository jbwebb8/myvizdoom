import tensorflow as tf
from network.layers import create_layer
import json

RESERVED_NAMES = ["state", "optimizer", "train_step"]

class NetworkBuilder:

    def __init__(self, network, network_file):
        self.network = network
        self.graph_dict = {}

    # Returns TensorFlow object with specified name in network file
    def _get_object(self, names):
        if isinstance(names, list):
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
        t = ph["kwargs"]["shape"] # for aesthetics

        # GameVariable input
        if len(t) == 1:
            ph["kwargs"]["shape"] = [None, t[0]]

        # Screen input
        # User specifies [H, W]
        elif len(t) == 2:
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
        try:
            is_training = self._get_object("is_training")
        except KeyError:
            is_training = None
        try:
            batch_size = self._get_object("batch_size")
        except KeyError:
            batch_size = None
        return create_layer(input_layer,
                            layer,
                            data_format=self.data_format,
                            is_training=is_training,
                            batch_size=batch_size)
    
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
        if not isinstance(params, list):
            params = [params]
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

        elif loss_type.lower() == "softmax_cross_entropy_with_logits":
            logits = prediction
            sum_axis = params[0]
            logits_shifted = logits - tf.reduce_max(logits, axis=sum_axis, keep_dims=True)
            log_sum_exp = tf.log(tf.reduce_sum(tf.exp(logits_shifted), axis=sum_axis, keep_dims=True)) # ln(Σe^x)
            xent = -tf.reduce_sum(tf.multiply(target, (logits_shifted - log_sum_exp)))
            tf.add_to_collection(tf.GraphKeys.LOSSES, xent)
            return xent

        ###########################################################
        # Add new loss function support here.
        # elif loss_type.lower() == "new_loss_fn":
        #     return <...>
        ###########################################################

        else:
            raise ValueError("Loss function \"" + loss_type + "\" not yet defined.")

    # Adds optimizer to graph
    def add_optimizer(self, opt_type, loss, var_list, params=None):
        """
        To ensure that, if batch normalization is present in any layer, the
        moving averages and variances are updated for each training step,
        we must prepend assignment of train_step with:

            update_ops = tf.GraphKeys.UPDATE_OPS
            with tf.control_dependencies(update_ops):
                train_step = ...
        
        """
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
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               scope=self.network.scope)
                with tf.control_dependencies(update_ops):
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
                    with tf.name_scope("gradients/mod"):
                        if layer_name.lower() == "all":
                            mod_gvs = [[modify_gradients(g, clip_type, *par[2:]), v]
                                    for g, v in mod_gvs]
                        else:
                            mod_gvs = [[modify_gradients(g, clip_type, *par[2:]), v]
                                    if layer_name in v.name else [g, v] for g, v in mod_gvs]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               scope=self.network.scope)
                with tf.control_dependencies(update_ops):
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
        elif net_type == "drqn":
            builder_type = _DRQN(self)
        elif net_type == "dueling_drqn":
            builder_type = _DuelingDRQN(self)
        elif net_type == "position":
            builder_type = _PositionEncoder(self)
        elif net_type == "decoder":
            builder_type = _Decoder(self)
        elif net_type == "custom":
            builder_type = _Custom(self)
        else:
            raise ValueError("Unknown network type: " + net_type)

        # Add placeholders
        builder_type._add_reserved_placeholders()
        self.graph_dict["state"] = [[], "p"]
        for ph in net["placeholders"]:
            if ph["name"] in net["global_features"]["input_layer"]:
                node = self.add_input_layer(ph)
                self.graph_dict["state"][0].append(node)
            else:
                node = self.add_placeholder(ph)
            self.graph_dict[ph["name"]] = [node, "p"]

        # Add layers
        self.graph_dict["rnn_states"] = []
        self.graph_dict["rnn_init_states"] = []
        for layer in net["layers"]:
            if layer["name"] in net["global_features"]["output_layer"]:
                l = builder_type._add_output_layer(layer)
            else:
                l = self.add_layer(layer)
            if isinstance(l, list): # recurrent layer
                self.graph_dict["rnn_states"].append(l[1])
                self.graph_dict["rnn_init_states"].append(l[2])
                l = l[0]
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
            if isinstance(loss_keys, list):
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
                                         var_list=var_list,
                                         params=opt_params)
            self.graph_dict["optimizer"] = [opt, "s"]
            self.graph_dict["train_step"] = [ts, "s"]
        else:
            if self.network.train_mode:
                raise ValueError("optimizer not found in network file.") 
        
        # Final check on use of reserved names
        for tf_type in ["placeholders", "layers", "ops"]:
            for n in net[tf_type]:
                if n["name"] in RESERVED_NAMES:
                    raise ValueError("Name \"" + n["name"] + "\" in " + tf_type + " is "
                                     + "reserved. Please choose different name.")


        return self.graph_dict, self.data_format           

    def add_summaries(self, var_list=None, val_list=None, loss_list=None):
        # Create summaries for trainable variables (weights and biases)
        var_sum = []
        if var_list is None:
            # Get all trainable variables
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.network.scope)
        with tf.name_scope("trainable_variables"):
            for var in var_list:
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
        if val_list is None:
            # Get all layers that are not empty lists
            val_list = self.graph_dict
            val_list = [v for v in val_list if (isinstance(val_list[v], list)
                                                and (len(val_list[v]) > 0)
                                                and (val_list[v][-1] == "l"))]
        with tf.name_scope("neurons"):
            for v in val_list:
                layer = self.graph_dict[v][0]
                with tf.name_scope(v):
                    num_elements = tf.cast(tf.size(layer, name="size"), tf.float64)
                    num_act = tf.cast(tf.count_nonzero(layer), tf.float64)
                    frac_act = tf.div(num_act, num_elements) # TODO: get fraction of neurons ever activated
                    neur_sum.append(tf.summary.scalar("frac_activated", frac_act))
                    neur_sum.append(tf.summary.histogram("values", layer))
        
        # Create summaries for gradients
        grad_sum = []
        if loss_list is None:
            # Get all losses classified under LOSSES GraphKey
            loss_list = tf.get_collection(tf.GraphKeys.LOSSES,
                                        scope=self.network.scope)
        optimizer = self.graph_dict["optimizer"][0]
        with tf.name_scope("gradients"):
            for i, l in enumerate(loss_list):
                gvs = optimizer.compute_gradients(l, var_list=var_list)
                for g, v in gvs:
                    with tf.name_scope(v.name[:-2]):
                        if g is not None:
                            grad_sum.append(tf.summary.histogram("grads_%d" % i, g))

        # Create summaries for losses
        loss_sum = []
        with tf.name_scope("losses"):
            for l in loss_list:
                mean_loss = tf.reduce_mean(l)
                loss_sum.append(tf.summary.scalar(l.name[:-2], mean_loss)) 

        return [s for s in [var_sum, neur_sum, grad_sum, loss_sum] if len(s) > 0]

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
        layer["kwargs"]["num_outputs"] = self.nb.network.num_outputs
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
            layer["kwargs"]["num_outputs"] = self.nb.network.num_outputs
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
            A_mean = tf.reduce_mean(A, axis=1, keep_dims=True)
            Q = tf.add(V, tf.subtract(A, A_mean), name="Q")
            self.nb.graph_dict["Q"] = [Q, "o"]

        # Add deterministic policy
        best_action = tf.argmax(self.nb.graph_dict["Q"][0], axis=1, 
                                name="best_action")
        self.nb.graph_dict["best_action"] = [best_action, "o"]

class _DRQN(_DQN):
    def __init__(self, network_builder):
        _DQN.__init__(self, network_builder)
    
    # Override DQN function to add batch_size
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
        self.nb.graph_dict["batch_size"] = [tf.placeholder(tf.int32,
                                                           shape=[],
                                                           name="batch_size"), "p"]
    
    # Override DQN function to apply loss mask
    def _add_loss_fn(self, loss_type, loss_params):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")
        w = tf.placeholder(tf.float32, shape=[None], name="IS_weights")

        # Create mask to ignore first mask_len losses in each trace
        w_ = w
        if "mask" in loss_params:
            idx = loss_params.index("mask")
            mask_len = loss_params.pop(idx+1)
            _ = loss_params.pop(idx)
            if not isinstance(mask_len, int) or mask_len < 0:
                raise ValueError("Length of loss mask must be nonnegative integer.")
            if mask_len > 0:
                with tf.name_scope("loss_mask"):
                    batch_size = self.nb.graph_dict["batch_size"][0]
                    tr_len = tf.shape(w)[0] // batch_size
                    mask_len = tf.minimum(mask_len, tr_len)
                    mask_zeros = tf.zeros([batch_size, mask_len])
                    mask_ones = tf.ones([batch_size, tr_len - mask_len])
                    w_ = tf.reshape(tf.concat([mask_zeros, mask_ones], axis=1), [-1])
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_type=loss_type,
                                    target=self.nb.graph_dict["target_q"][0],
                                    prediction=q_sa,
                                    weights=w_,
                                    params=loss_params)
        self.nb.graph_dict["IS_weights"] = [w, "p"]
        self.nb.graph_dict["loss"] = [loss_fn, "o"]


class _DuelingDRQN(_DuelingDQN):
    def __init__(self, network_builder):
        _DuelingDQN.__init__(self, network_builder)
    
    # Override DQN function to add batch_size
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
        self.nb.graph_dict["batch_size"] = [tf.placeholder(tf.int32,
                                                           shape=[],
                                                           name="batch_size"), "p"]
    
    # Override DQN function to apply loss mask
    def _add_loss_fn(self, loss_type, loss_params):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")
        w = tf.placeholder(tf.float32, shape=[None], name="IS_weights")

        # Create mask to ignore first mask_len losses in each trace
        w_ = w
        if "mask" in loss_params:
            idx = loss_params.index("mask")
            mask_len = loss_params.pop(idx+1)
            _ = loss_params.pop(idx)
            if not isinstance(mask_len, int) or mask_len < 0:
                raise ValueError("Length of loss mask must be nonnegative integer.")
            if mask_len > 0:
                with tf.name_scope("loss_mask"):
                    batch_size = self.nb.graph_dict["batch_size"][0]
                    tr_len = tf.shape(w)[0] // batch_size
                    mask_len = tf.minimum(mask_len, tr_len)
                    mask_zeros = tf.zeros([batch_size, mask_len])
                    mask_ones = tf.ones([batch_size, tr_len - mask_len])
                    w_ = tf.reshape(tf.concat([mask_zeros, mask_ones], axis=1), [-1])
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_type=loss_type,
                                    target=self.nb.graph_dict["target_q"][0],
                                    prediction=q_sa,
                                    weights=w_,
                                    params=loss_params)
        self.nb.graph_dict["IS_weights"] = [w, "p"]
        self.nb.graph_dict["loss"] = [loss_fn, "o"]

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
            layer["kwargs"]["num_outputs"] = self.nb.network.num_outputs
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

class _PositionEncoder:

    def __init__(self, network_builder):
        self.nb = network_builder

    def _add_reserved_placeholders(self):
        self.nb.graph_dict["position"] = [tf.placeholder(tf.float32, 
                                                        shape=[None, 2],
                                                        name="position"), "p"]
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                       shape=[None, 2], 
                                                       name="IS_weights"), "p"]
    
    def _add_output_layer(self, layer):
        layer["name"] = "POS"
        layer["kwargs"]["num_outputs"] = 2
        return self.nb.add_layer(layer)
    
    def _add_reserved_ops(self):
        pass
    
    def _add_loss_fn(self, loss_type, loss_params):
        target = self.nb.graph_dict["position"][0]
        prediction = self.nb.graph_dict["POS"][0]
        w = self.nb.graph_dict["IS_weights"][0]
        loss_fn = self.nb.add_loss_fn(loss_type, target, prediction, 
                            weights=w, params=loss_params)
        self.nb.graph_dict["IS_weights"] = [w, "p"]
        self.nb.graph_dict["loss"] = [loss_fn, "o"]

class _Decoder:

    def __init__(self, network_builder):
        self.nb = network_builder

    def _add_reserved_placeholders(self):
        self.nb.graph_dict["target_pos"] = [tf.placeholder(tf.float32, 
                                                           shape=[None, 2],
                                                           name="position"), "p"]
        self.nb.graph_dict["target_act"] = [tf.placeholder(tf.int32, 
                                                           shape=[None], 
                                                           name="action"), "p"]
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                           shape=[None], 
                                                           name="IS_weights"), "p"]
    
    def _add_output_layer(self, layer):
        if layer["name"].lower() in ["pos", "position"]:
            layer["name"] = "POS"
            layer["num_outputs"] = 2
        elif layer["name"].lower() == "r":
            layer["name"] = "R"
            layer["num_outputs"] = 1
        elif layer["name"].lower() in ["act", "actions"]:
            layer["name"] = "ACT_logits"
            #layer["num_outputs"] = self.nb.network.encoding_net.num_outputs
        return self.nb.add_layer(layer)
    
    def _add_reserved_ops(self):
        # Create softmax for action prediction (logits needed for stable loss fn)
        if "ACT_logits" in self.nb.graph_dict:
            act_logits = self.nb.graph_dict["ACT_logits"][0]
            self.nb.graph_dict["ACT_softmax"] = tf.nn.softmax(act_logits)
    
    def _add_loss_fn(self, loss_type, loss_params):
        w = tf.placeholder(tf.float32, shape=[None], name="IS_weights")
        loss_list = []

        # Get loss types and params
        if loss_type.lower() == "standard":
            loss_type_pos = "mean_squared_error"
            loss_params_pos = None
            loss_type_r = "mean_squared_error"
            loss_params_r = None
            loss_type_act = "softmax_cross_entropy_with_logits"
            loss_params_act = 1 # axis to sum over
        else:
            raise ValueError("Loss type \"" + loss_type 
                             + "\" not supported for Decoder class.")
        
        # Position loss: default is MSE
        if "POS" in self.nb.graph_dict:
            with tf.name_scope("loss_pos"):
                pred_pos = self.nb.graph_dict["POS"][0]
                pred_len = pred_pos.get_shape().as_list()[1]
                target_pos = self.nb.graph_dict["target_pos"][0]
                target_pos = tf.reshape(target_pos, shape=[-1, pred_len, 2])
                w_ = tf.expand_dims(tf.expand_dims(w, axis=1), axis=2)
                loss_pos = self.nb.add_loss_fn(loss_type=loss_type_pos, 
                                            target=target_pos, 
                                            prediction=pred_pos, 
                                            weights=w_, 
                                            params=loss_params_pos)
                self.nb.graph_dict["loss_pos"] = [loss_pos, "o"]
                loss_list.append(loss_pos)
        
        # Radius loss: default is MSE
        if "R" in self.nb.graph_dict:
            with tf.name_scope("loss_r"):
                pred_r = self.nb.graph_dict["R"][0]
                pred_len = pred_r.get_shape().as_list()[1]
                if "POS" not in self.nb.graph_dict:
                    target_pos = self.nb.graph_dict["target_pos"][0]
                    target_pos = tf.reshape(target_pos, shape=[-1, pred_len, 2])
                target_r = tf.sqrt(tf.reduce_sum(tf.square(target_pos), 
                                                 axis=2, 
                                                 keep_dims=True),
                                   name="target_r")
                w_ = tf.expand_dims(tf.expand_dims(w, axis=1), axis=2)
                loss_r = self.nb.add_loss_fn(loss_type=loss_type_r,
                                             target=target_r,
                                             prediction=pred_r,
                                             weights=w_,
                                             params=loss_params_r)
                self.nb.graph_dict["loss_r"] = [loss_r, "o"]
                loss_list.append(loss_r)
        
        # Action loss: default is cross-entropy
        if "ACT_logits" in self.nb.graph_dict:
            with tf.name_scope("loss_act"):
                pred_act = self.nb.graph_dict["ACT_logits"][0]
                pred_len = pred_act.get_shape().as_list()[1]
                target_act = self.nb.graph_dict["target_act"][0]
                target_act_rs = tf.reshape(target_act, shape=[-1, pred_len])
                target_act_oh = tf.one_hot(target_act_rs, self.nb.network.encoding_net.num_outputs)
                w_ = tf.expand_dims(w, axis=1)
                loss_act = self.nb.add_loss_fn(loss_type=loss_type_act, 
                                            target=target_act_oh, 
                                            prediction=pred_act, 
                                            weights=w_, 
                                            params=loss_params_act)
                self.nb.graph_dict["loss_act"] = [loss_act, "o"]
                loss_list.append(loss_act)

        self.nb.graph_dict["IS_weights"] = [w, "p"]
        loss_tot = tf.add_n(loss_list, name="loss_tot")
        self.nb.graph_dict["loss_tot"] = [loss_tot, "o"]

# TODO: give stand-alone capability to JSON file building
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

# Do not use the class below. It is just a template for new class creation.
# If you want to build the network entirely from the JSON file, then use the
# _Custom class
class _Template:

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