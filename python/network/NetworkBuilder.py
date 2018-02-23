import tensorflow as tf
from network.layers import create_layer
from network.ops import create_op
from network.losses import create_loss_fn
from network.optimizers import create_optimizer, create_train_step
from utils import recursive_dict_search
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

    def _add_feature(self, feature_type, feature):
        if feature_type.lower() == "placeholders":
            return self.add_placeholder(feature)
        elif feature_type.lower() == "layers":
            return self.add_layer(feature)
        elif feature_type.lower() == "ops":
            return self.add_op(feature)
        elif feature_type.lower() == "losses":
            return self.add_loss_fn(feature)
        elif feature_type.lower() == "optimizers":
            return self.add_optimizer(feature)
        elif feature_type.lower() == "train_steps":
            return self.add_train_step(feature)
        else:
            raise ValueError("Unknown feature type \"" + feature_type + "\".")

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
        # Below should be covered by recursive_dict_search
        #if "shape" in ph["kwargs"]:
        #    for i in range(len(ph["kwargs"]["shape"])):
        #        if ph["kwargs"]["shape"][i] == "None":
        #            ph["kwargs"]["shape"][i] = None
        return tf.placeholder(ph["data_type"], **ph["kwargs"])
    
    # Adds layer to graph
    def add_layer(self, layer={}, **kwargs):
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
                            batch_size=batch_size,
                            **kwargs)
    
    # Adds operation to graph
    def add_op(self, op={}, **kwargs):
        # Replace references in input list with graph_dict objects
        try:
            input_op = self._get_object(op["input"])
        except KeyError:
            input_op = []
        
        # Replace references in keyword args with graph_dict objects
        for k, v in op["kwargs"].items():
            if v in self.graph_dict:
                op["kwargs"][k] = self._get_object(v)
        
        return create_op(input_op, op, **kwargs)

    # Adds loss function to graph
    def add_loss_fn(self, loss={}, **kwargs):
        # Replace references in input list with graph_dict objects
        try:
            input_loss = self._get_object(loss["input"])
        except KeyError:
            input_loss = []
        
        # Replace references in keyword args with graph_dict objects
        for k, v in loss["kwargs"].items():
            if v in self.graph_dict:
                loss["kwargs"][k] = self._get_object(v)

        return create_loss_fn(input_loss, loss, **kwargs)

    # Adds optimizer to graph
    def add_optimizer(self, opt={}, **kwargs):
        return create_optimizer(opt, **kwargs)

    # Adds train step from an optimizer to graph
    def add_train_step(self, ts={}, **kwargs):
        # Get optimizer object
        opt = self._get_object(ts["optimizer"])
        ts["optimizer"] = opt

        return create_train_step(ts, **kwargs)

    def load_json(self, network_file):
        # Load arguments from network file
        net = json.loads(open(network_file).read())
        self._set_data_format(net)

        # Replace "None" string with None value
        recursive_dict_search(net, "None", None)

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

        if net_type == "custom":
            return self._build_custom_json(net)
        else:
            return self._build_preset_json(net, builder_type)

    def _build_preset_json(self, net, builder_type):
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
        
        # Add loss functions
        for loss in net["losses"]:
            if loss["name"] in net["global_features"]["loss"]:
                l = builder_type._add_loss_fn(loss)
            else:
                l = self.add_loss_fn(loss)
            
            # This may lead to double storage of the RL loss function if under
            # a different name in the JSON file, but that's okay.
            self.graph_dict[loss["name"]] = [l, "o"]

        if self.network.train_mode and loss["name"] not in net["global_features"]: 
            raise ValueError("Main loss fn not found in network file.")
        
        # Add optimizers
        for opt in net["optimizers"]:
            if self.network.learning_rate is not None:
                try:
                    opt["kwargs"]["learning_rate"] = self.network.learning_rate
                except KeyError:
                    opt["kwargs"] = {"learning_rate": self.network.learning_rate}
            optimizer = self.add_optimizer(opt)
            self.graph_dict[opt["name"]] = [optimizer, "s"]
        if self.network.train_mode and len(net["optimizers"]) == 0:
            raise ValueError("optimizer not found in network file.")    

        # Create train steps
        train_steps = []
        for ts in net["train_steps"]:
            train_step = self.add_train_step(ts)
            train_steps.append(train_step)
            self.graph_dict[ts["name"]] = [train_step, "s"]
        self.graph_dict["train_step"] = train_steps + ["s"]
        if self.network.train_mode and len(net["train_steps"]) == 0:
            raise ValueError("optimizer not found in network file.") 
        
        # Final check on use of reserved names
        for tf_type in ["placeholders", "layers", "ops"]:
            for n in net[tf_type]:
                if n["name"] in RESERVED_NAMES:
                    raise ValueError("Name \"" + n["name"] + "\" in " + tf_type + " is "
                                     + "reserved. Please choose different name.")

        return self.graph_dict, self.data_format           

    def _build_custom_json(self, net):
        # Add features in order of dependence (brute force method).
        # Can try smarter algorithm later if have time. If really have
        # extra time, probably could incorporate graph theory here.
        
        # Initial run-through, saving out-of-order features
        feat_list = []
        i = 0
        for feat_type, feats in net.items():
            if feat_type in ["global_features"]:
                continue
            for feat in feats:
                try:
                    node = self._add_feature(feat_type, feat)
                    if not isinstance(node, list):
                        node = [node]
                    name = feat.get("name", "feature_%d" % i)
                    self.graph_dict[name] = node + [feat_type] # covers RNNs
                    i += 1
                except (KeyError, TypeError) as e:
                    feat_list.append([feat_type, feat])
        
        # Continuous run-through, looping until all features built
        while len(feat_list) > 0:
            start_len = len(feat_list)
            for j, [feat_type, feat] in enumerate(feat_list):
                try:
                    node = self._add_feature(feat_type, feat)
                    if not isinstance(node, list):
                        node = [node]
                    name = feat.get("name", "feature_%d" % i)
                    self.graph_dict[name] = node + [feat_type] # covers RNNs
                    i += 1
                    _ = feat_list.pop(j)
                except (KeyError, TypeError):
                    pass
            if start_len == len(feat_list):
                msg = (', ').join([str(feat.get("name", "unnamed" + feat_type)) 
                                   for feat_type, feat in feat_list])
                raise SyntaxError("Features " + msg + " could not be added to graph.")

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
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                shape=[None], 
                                                name="IS_weights"), "p"]

    def _add_output_layer(self, layer):
        layer["name"] = "Q"
        layer["kwargs"]["num_outputs"] = self.nb.network.num_outputs
        return self.nb.add_layer(layer)                

    def _add_reserved_ops(self):
        best_action = tf.argmax(self.nb.graph_dict["Q"][0], axis=1, 
                                name="best_action")
        self.nb.graph_dict["best_action"] = [best_action, "o"]
    
    def _add_loss_fn(self, loss_dict):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_dict,
                                      target=self.nb.graph_dict["target_q"][0],
                                      prediction=q_sa,
                                      weights=self.nb.graph_dict["IS_weights"][0])
        self.nb.graph_dict["loss"] = [loss_fn, "o"]

        return loss_fn

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
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                shape=[None], 
                                                name="IS_weights"), "p"]
        self.nb.graph_dict["batch_size"] = [tf.placeholder(tf.int32,
                                                           shape=[],
                                                           name="batch_size"), "p"]

    # Override DQN function to apply loss mask
    def _add_loss_fn(self, loss_dict):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")

        # Create mask to ignore first mask_len losses in each trace
        if "mask" in loss_dict:
            w = self.nb.add_op(op_type="rnn_loss_mask",
                                weights=self.nb.graph_dict["IS_weights"][0],
                                mask_len=loss_dict["mask"],
                                batch_size=self.nb.graph_dict["batch_size"][0])
        else:
            w = self.nb.graph_dict["IS_weights"][0]
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_dict,
                                      target=self.nb.graph_dict["target_q"][0],
                                      prediction=q_sa,
                                      weights=w)
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
        self.nb.graph_dict["IS_weights"] = [tf.placeholder(tf.float32, 
                                                shape=[None], 
                                                name="IS_weights"), "p"]
        self.nb.graph_dict["batch_size"] = [tf.placeholder(tf.int32,
                                                           shape=[],
                                                           name="batch_size"), "p"]
    
    def _add_loss_fn(self, loss_dict):
        # Extract Q(s,a) and utilize importance sampling weights
        q_sa = tf.gather_nd(self.nb.graph_dict["Q"][0], 
                            self.nb.graph_dict["actions"][0], 
                            name="q_sa")

        # Create mask to ignore first mask_len losses in each trace
        if "mask" in loss_dict:
            w = self.nb.add_op(op_type="rnn_loss_mask",
                                weights=self.nb.graph_dict["IS_weights"][0],
                                mask_len=loss_dict["mask"],
                                batch_size=self.nb.graph_dict["batch_size"][0])
        else:
            w = self.nb.graph_dict["IS_weights"][0]
    
        # Add loss function
        loss_fn = self.nb.add_loss_fn(loss_dict,
                                      target=self.nb.graph_dict["target_q"][0],
                                      prediction=q_sa,
                                      weights=w)
        self.nb.graph_dict["loss"] = [loss_fn, "o"]

### Not configured with new format ###
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
        
        self.nb.graph_dict["loss_pi"] = [pi_loss_fn, "o"]
        self.nb.graph_dict["loss_v"] = [v_loss_fn, "o"]

### Not configured with new format ###
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
    LOSS_OUTPUTS = ["position", "r", "action"]

    def __init__(self, network_builder):
        self.nb = network_builder
        self.loss_list = []

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
    
    def _add_loss_fn(self, loss_dict):
        # Get IS weights and loss output type
        w = self.nb.graph_dict["IS_weights"][0]
        try:
            loss_output = loss_dict["output"]
        except KeyError:
            raise SyntaxError("Loss output must be provided.")

        # Position loss
        if loss_output.lower() == "position":
            with tf.name_scope("loss_pos"):
                pred_pos = self.nb.graph_dict["POS"][0]
                pred_len = pred_pos.get_shape().as_list()[1]
                target_pos = self.nb.graph_dict["target_pos"][0]
                target_pos = tf.reshape(target_pos, shape=[-1, pred_len, 2])
                w_ = tf.expand_dims(tf.expand_dims(w, axis=1), axis=2)
                loss_pos = self.nb.add_loss_fn(loss_dict, 
                                               target=target_pos, 
                                               prediction=pred_pos, 
                                               weights=w_)
                self.nb.graph_dict["loss_pos"] = [loss_pos, "o"]
                self.loss_list.append(loss_pos)
        
        # Radius loss
        if loss_output.lower() == "r":
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
                loss_r = self.nb.add_loss_fn(loss_dict,
                                             target=target_r,
                                             prediction=pred_r,
                                             weights=w_)
                self.nb.graph_dict["loss_r"] = [loss_r, "o"]
                self.loss_list.append(loss_r)
        
        # Action loss: default is cross-entropy
        if loss_output.lower() == "action":
            with tf.name_scope("loss_act"):
                pred_act = self.nb.graph_dict["ACT_logits"][0]
                pred_len = pred_act.get_shape().as_list()[1]
                target_act = self.nb.graph_dict["target_act"][0]
                target_act_rs = tf.reshape(target_act, shape=[-1, pred_len])
                target_act_oh = tf.one_hot(target_act_rs, self.nb.network.encoding_net.num_outputs)
                w_ = tf.expand_dims(w, axis=1)
                loss_act = self.nb.add_loss_fn(loss_dict, 
                                               target=target_act_oh, 
                                               prediction=pred_act, 
                                               weights=w_)
                self.nb.graph_dict["loss_act"] = [loss_act, "o"]
                self.loss_list.append(loss_act)

        # Add all losses in loss_list to create single total_loss node.
        # This will update the graph_dict every time the function is called,
        # but the final value will represent all loss functions.
        loss_tot = tf.add_n(loss_list, name="loss_tot")
        self.nb.graph_dict["loss_tot"] = [loss_tot, "o"]

# Do not use the class below. It is just a template for new class creation.
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
    
    def _add_loss_fn(self, loss_dict):
        target = self.nb.graph_dict["target"]
        prediction = self.nb.graph_dict[self.output_name]
        self.nb.add_loss_fn(loss_dict,
                            target=target, 
                            prediction=prediction, 
                            weights=None)