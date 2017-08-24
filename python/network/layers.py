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
            return arg[0], [None]
    else:
        return arg, [None]

def _get_variable_initializer(init_type, var_shape, *args):
    if init_type == "random_normal":
        mean = float(args[0])
        stddev = float(args[1])
        return tf.random_normal(var_shape, mean=mean, stddev=stddev)
    elif init_type == "constant":
        c = args[0]
        return tf.constant(c, dtype=tf.float32, shape=var_shape)

def _apply_normalization(norm_type, x, *args, **kwargs):
    if norm_type == "batch_norm":
        return batch_norm(x, *args, **kwargs)

def _apply_activation(activation_type, x, *args):
    if activation_type.lower() == "relu":
        return tf.nn.relu(x, name="Relu")
    elif activation_type.lower() == "softmax":
        return tf.nn.softmax(x)

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
        # Create weights
        W_init_type, W_init_params = _check_list(weights_initializer)
        if data_format == "NCHW":
            input_channels = input_layer.get_shape().as_list()[1]
        elif data_format == "NHWC":
            input_channels = input_layer.get_shape().as_list()[3]
        W_shape = kernel_size + [input_channels, num_outputs]
        W_init = _get_variable_initializer(W_init_type,
                                            W_shape,
                                            *W_init_params)
        W = tf.Variable(W_init, 
                        dtype=tf.float32, 
                        trainable=trainable, 
                        name="weights")
        

        # Convolute input
        stride_h, stride_w = _check_list(stride)
        if stride_w is None: stride_w = stride_h
        elif isinstance(stride_w, list): stride_w = stride_w[0]
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
        # Create weights
        W_init_type, W_init_params = _check_list(weights_initializer)
        W_shape = [input_layer.get_shape().as_list()[1], num_outputs]
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
            pass

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
               scope="BatchNorm"):
    """
    norm_dim: Breadth of normalization to perform.
    - global: Normalize channel-by-channel over batch size, height, and width
    - simple: Normalize neuron-by-neuron over batch size
    """
    with tf.name_scope(scope):
        # Create trainable variables γ and β, and running population stats
        gamma = tf.Variable(tf.ones(1))
        beta = tf.Variable(tf.zeros(1))
        pop_mean = tf.Variable(tf.zeros(1), trainable=False)
        pop_var = tf.Variable(tf.ones(1), trainable=False)

        # Get shape of input; assume shape [batch_size, ...]
        if data_format is None:
            mom_axes = [0] # simple batch normalization

        # Apply batch normalizing transform to x:
        # x_hat = (x - μ)/(σ^2 + ε)^0.5
        # y_hat = γ * x_hat + β

        # If training, use batch statistics for transformation and update
        # population statistics via moving average and variance.
        batch_mean, batch_var = tf.nn.moments(x, mom_axes, name="moments")
        x_hat_train = (x - batch_mean) / (tf.sqrt(batch_var + epsilon))
        new_pop_mean = (1 - decay) * batch_mean + decay * pop_mean
        new_pop_var = (1 - decay) * batch_var + decay * pop_var
        update_pop_mean = tf.assign(pop_mean, new_pop_mean)
        update_pop_var = tf.assign(pop_var, new_pop_var)
        
        # If testing, use population statistics for transformation
        x_hat_test = (x - pop_mean) / (tf.sqrt(pop_var + epsilon))
        
        # Apply learned linear transformation to transformed input based
        # on phase
        is_training = tf.placeholder(tf.bool, name="is_training")
        x_hat = tf.cond(is_training, x_hat_train, x_hat_test)
        out = gamma * x_hat + beta
        return out