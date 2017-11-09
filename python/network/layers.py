import tensorflow as tf

def create_layer(input_layer, layer_dict, data_format="NHWC"):
    layer_type = layer_dict["type"]
    if layer_type.lower() == "conv2d":
        layer_dict["kwargs"]["data_format"] = data_format
        return conv2d(input_layer, **layer_dict["kwargs"])
    elif layer_type.lower() == "flatten":
        layer_dict["kwargs"]["data_format"] = data_format
        return flatten(input_layer, **layer_dict["kwargs"])
    elif layer_type.lower() == "fully_connected":
        return fully_connected(input_layer, **layer_dict["kwargs"])
    elif layer_type.lower() == "multi_input_fully_connected":
        return multi_input_fully_connected(input_layer, **layer_dict["kwargs"])
    else:
        raise ValueError("Layer type \"" + layer_type + "\" not supported.")

def _assign_kwargs(layer_kwargs):
    # Format:
    # if _ in layer_kwargs:
    #     if _ == _:
    #         return _
    pass

def _check_list(arg):
    if isinstance(arg, list):
        try:
            return arg[0], arg[1:]
        except IndexError:
            return arg[0], []
    else:
        return arg, []

def _get_variable_initializer(init_type, var_shape, *args):
    if init_type == "random_normal":
        mean = float(args[0])
        stddev = float(args[1])
        return tf.random_normal(var_shape, mean=mean, stddev=stddev)
    elif init_type == "truncated_normal":
        mean = float(args[0])
        stddev = float(args[1])
        return tf.truncated_normal(var_shape, mean=mean, stddev=stddev)
    elif init_type == "constant":
        c = args[0]
        return tf.constant(c, dtype=tf.float32, shape=var_shape)
    elif init_type == "xavier":
        n_in = tf.cast(args[0], tf.float32)
        return tf.div(tf.random_normal(var_shape), tf.sqrt(n_in))
    else:
        raise ValueError("Variable initializer \"" + init_type + "\" not supported.")

def _apply_normalization(norm_type, x, *args, **kwargs):
    if norm_type == "batch_norm":
        return batch_norm(x, *args, **kwargs)
    else:
        raise ValueError("Normalization type \"" + norm_type + "\" not supported.")

def _apply_activation(activation_type, x, *args):
    if activation_type.lower() == "relu":
        return tf.nn.relu(x, name="Relu")
    elif activation_type.lower() == "leaky_relu":
        return tf.maximum(x, 0.1 * x, name="Leaky_Relu")
    elif activation_type.lower() == "softmax":
        return tf.nn.softmax(x)
    elif activation_type.lower() == "none":
        return x
    else:
        raise ValueError("Activation type \"" + activation_type + "\" not supported.")

def conv2d(input_layer,
           num_outputs,
           kernel_size,
           stride=1,
           padding="VALID",
           data_format="NCHW",
           normalizer_fn=None,
           activation_fn=None,
           weights_initializer="random_normal",
           biases_initializer=None,
           trainable=True,
           scope="CONV"):
    with tf.name_scope(scope):
        input_shape = input_layer.get_shape().as_list()
        
        # Create weights
        W_init_type, W_init_params = _check_list(weights_initializer)
        with tf.name_scope(W_init_type + "_initializer"):
            if data_format == "NHWC":
                input_channels = input_shape[3]
            elif data_format == "NCHW":
                input_channels = input_shape[1]
            W_shape = kernel_size + [input_channels, num_outputs]
            if W_init_type == "xavier":
                layer_shape = input_shape[1:]
                n_in = tf.reduce_prod(layer_shape)
                W_init_params = [n_in] 
            W_init = _get_variable_initializer(W_init_type,
                                                W_shape,
                                                *W_init_params)
        W = tf.Variable(W_init, 
                        dtype=tf.float32, 
                        trainable=trainable, 
                        name="weights")
        

        # Convolute input
        stride_h, stride_w = _check_list(stride)
        if isinstance(stride_w, list):
            if len(stride_w) == 0:
                stride_w = stride_h
            else:
                stride_w = stride_w[0]
        if data_format == "NHWC":
            strides = [1, stride_h, stride_w, 1]
        elif data_format == "NCHW":
            strides = [1, 1, stride_h, stride_w]
        out = tf.nn.conv2d(input_layer, 
                            filter=W,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                            name="convolution")
        
        # Apply normalization
        if normalizer_fn is not None:
            norm_type, norm_params = _check_list(normalizer_fn)
            out = _apply_normalization(norm_type, 
                                       out, 
                                       *norm_params,
                                       data_format=data_format)
        
        # Add biases
        elif biases_initializer is not None:
            b_init_type, b_init_params = _check_list(biases_initializer)
            if data_format == "NHWC":
                b_shape = [1, 1, 1, num_outputs]
            elif data_format == "NCHW":
                b_shape = [1, num_outputs, 1, 1]
            b_init = _get_variable_initializer(b_init_type,
                                               b_shape,
                                               *b_init_params)
            b = tf.Variable(b_init,
                            dtype=tf.float32,
                            trainable=trainable,
                            name="biases")
            out = tf.add(out, b, name="BiasAdd")

        # Apply activation
        if activation_fn is not None:
            act_type, act_params = _check_list(activation_fn)
            out = _apply_activation(act_type, out, *act_params)

        return out

def flatten(input_layer, 
            data_format="NCHW",
            scope="FLAT"):
    with tf.name_scope(scope):
        # Yes, I basically copied tf.contrib.layers.flatten, but
        # it was a good learning experience!
        # Grab runtime values to determine number of elements
        input_shape = tf.shape(input_layer)
        input_ndims = input_layer.get_shape().ndims
        batch_size = tf.slice(input_shape, [0], [1])
        layer_shape = tf.slice(input_shape, [1], [input_ndims-1])
        num_neurons = tf.expand_dims(tf.reduce_prod(layer_shape), 0)
        flattened_shape = tf.concat([batch_size, num_neurons], 0)
        if data_format == "NHWC":
            input_layer = tf.transpose(input_layer, perm=[0, 3, 1, 2])
        flat = tf.reshape(input_layer, flattened_shape)
        
        # Attempt to set values during graph building
        input_shape = input_layer.get_shape().as_list()
        batch_size, layer_shape = input_shape[0], input_shape[1:]
        if all(layer_shape): # None not present
            num_neurons = 1
            for dim in layer_shape:
                num_neurons *= dim
            flat.set_shape([batch_size, num_neurons])
        else: # None present
            flat.set_shape([batch_size, None])
        return flat

def fully_connected(input_layer,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=None,
                    weights_initializer="random_normal",
                    biases_initializer=None,
                    trainable=True,
                    scope="FC"):
    with tf.name_scope(scope):
        input_shape = input_layer.get_shape().as_list()
        
        # Create weights
        W_init_type, W_init_params = _check_list(weights_initializer)
        with tf.name_scope(W_init_type + "_initializer"):
            W_shape = [input_shape[1], num_outputs]
            if W_init_type == "xavier":
                layer_shape = input_shape[1]
                n_in = tf.reduce_prod(layer_shape)
                W_init_params = [n_in]
            W_init = _get_variable_initializer(W_init_type,
                                            W_shape,
                                            *W_init_params)
        W = tf.Variable(W_init,
                        dtype=tf.float32, 
                        trainable=trainable, 
                        name="weights")
        
        # Multiply inputs by weights
        out = tf.matmul(input_layer, W)

        # Apply normalization
        if normalizer_fn is not None:
            norm_type, norm_params = _check_list(normalizer_fn)
            out = _apply_normalization(norm_type, 
                                       out, 
                                       *norm_params,
                                       data_format=None)

        # Add biases
        elif biases_initializer is not None:
            b_init_type, b_init_params = _check_list(biases_initializer)
            b_shape = [num_outputs]
            b_init = _get_variable_initializer(b_init_type,
                                               b_shape,
                                               *b_init_params)
            b = tf.Variable(b_init,
                            dtype=tf.float32,
                            trainable=trainable,
                            name="biases")
            out = tf.add(out, b, name="BiasAdd")
       
        # Apply activation
        if activation_fn is not None:
            act_type, act_params = _check_list(activation_fn)
            out = _apply_activation(act_type, out, *act_params)

        return out

def multi_input_fully_connected(input_layers,
                                num_outputs,
                                normalizer_fn=None,
                                activation_fn=None,
                                weights_initializer="random_normal",
                                biases_initializer=None,
                                trainable=True,
                                scope="FC"):
    with tf.name_scope(scope):
        affine = []
        for idx, layer in enumerate(input_layers):
            affine.append(fully_connected(layer,
                                          num_outputs,
                                          normalizer_fn=None,
                                          activation_fn=None,
                                          weights_initializer=weights_initializer,
                                          biases_initializer=None,
                                          trainable=trainable,
                                          scope=scope + "_" + str(idx)))
        out = tf.add_n(affine, name="add_affine")

        # Apply normalization
        if normalizer_fn is not None:
            norm_type, norm_params = _check_list(normalizer_fn)
            out = _apply_normalization(norm_type, 
                                       out, 
                                       *norm_params,
                                       data_format=None)

        # Add biases
        elif biases_initializer is not None:
            b_init_type, b_init_params = _check_list(biases_initializer)
            b_shape = [num_outputs]
            b_init = _get_variable_initializer(b_init_type,
                                               b_shape,
                                               *b_init_params)
            b = tf.Variable(b_init,
                            dtype=tf.float32,
                            trainable=trainable,
                            name="biases")
            out = tf.add(out, b, name="BiasAdd")
        
        # Apply activation
        if activation_fn is not None:
            act_type, act_params = _check_list(activation_fn)
            out = _apply_activation(act_type, out, *act_params)

        return out

def batch_norm(x,
               decay=0.999,
               epsilon=0.001,
               data_format=None,
               norm_dim="global",
               is_training=None,
               scope="BatchNorm"):
    """
    norm_dim: Breadth of normalization to perform.
    - global: Normalize channel-by-channel over batch size, height, and width
    - simple: Normalize neuron-by-neuron over batch size
    """
    with tf.name_scope(scope):
        # Get shape of input; assume shape [batch_size, ...]
        if data_format is None or norm_dim == "simple":
            mom_axes = [0] # simple batch normalization
            param_shape = [1, -1]
            num_channels = x.get_shape().as_list()[1:]
        elif data_format == "NCHW":
            mom_axes = [0, 2, 3]
            param_shape = [1, -1, 1, 1]
            num_channels = x.get_shape().as_list()[1]
        elif data_format == "NHWC":
            mom_axes = [0, 1, 2]
            param_shape = [1, 1, 1, -1]
            num_channels = x.get_shape().as_list()[3]
        else:
            raise SyntaxError("Unknown data format:", data_format)

        # Create trainable variables γ and β, and running population stats
        gamma = tf.Variable(tf.ones(num_channels), name="gamma")
        beta = tf.Variable(tf.zeros(num_channels), name="beta")
        pop_mean = tf.Variable(tf.zeros(num_channels), trainable=False, name="pop_mean")
        pop_var = tf.Variable(tf.ones(num_channels), trainable=False, name="pop_var")
        
        # Try to grab is_training placeholder from graph if already exists
        # Otherwise create is_training placeholder
        is_training = is_training
        if is_training is None:
            for op in tf.get_default_graph().get_operations():
                if op.type == "Placeholder" and op.name.endswith("is_training"):
                    is_training = tf.get_default_graph().get_tensor_by_name(op.name + ":0")
                    break
            if is_training is None:
                is_training = tf.placeholder(tf.bool, name="is_training")
                print("Warning: No placeholder found for \"is_training\". "
                      + "One has been automatically created but may not be " 
                      + "passed to the Agent object. Consider adding "
                      + "placeholder if using JSON file.")

        # Apply batch normalizing transform to x:
        # x_hat = (x - μ)/(σ^2 + ε)^0.5
        # y_hat = γ * x_hat + β

        # If training, use batch statistics for transformation and update
        # population statistics via moving average and variance.
        batch_mean, batch_var = tf.nn.moments(x, mom_axes, name="moments")
        with tf.name_scope("update_pop_mean"):
            new_pop_mean = (1.0 - decay) * batch_mean + decay * pop_mean
            update_pop_mean = tf.assign(pop_mean, new_pop_mean)
        with tf.name_scope("update_pop_var"):
            new_pop_var = (1.0 - decay) * batch_var + decay * pop_var
            update_pop_var = tf.assign(pop_var, new_pop_var)
        def x_hat_train():
            with tf.control_dependencies([update_pop_mean, update_pop_var]):
                with tf.name_scope("batch_stats"):
                    batch_mean_rs = tf.reshape(batch_mean, param_shape)
                    batch_var_rs = tf.reshape(batch_var, param_shape)
                    return (x - batch_mean_rs) / (tf.sqrt(batch_var_rs + epsilon))
        
        # If testing, use population statistics for transformation
        def x_hat_test():
            with tf.name_scope("pop_stats"):
                pop_mean_rs = tf.reshape(pop_mean, param_shape)
                pop_var_rs = tf.reshape(pop_var, param_shape)
                return (x - pop_mean_rs) / (tf.sqrt(pop_var_rs + epsilon))

        # Apply learned linear transformation to transformed input based
        # on phase
        x_hat = tf.cond(is_training, 
                        x_hat_train, 
                        x_hat_test)
        gamma_rs = tf.reshape(gamma, param_shape)
        beta_rs = tf.reshape(beta, param_shape)
        out = gamma_rs * x_hat + beta_rs

        return out