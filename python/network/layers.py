import tensorflow as tf

def create_layer(layer_dict, input_layer):
    # Format:
    # if layer_name == "name":
    #     return create_<layer_name>(input_layer, **layer_dict)
    layer_type = layer_dict["type"]
    if layer_type.lower() == "conv2d":
        
    pass

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
            return arg[0]
    else:
        return arg, None

def _get_variable_initializer(init_type, var_shape, *args):
    if init_type == "random_normal":
        mean = float(args[0])
        stddev = float(args[1])
        return tf.random_normal(var_shape, mean=mean, stddev=stddev)
    elif init_type == "constant":
        c = args[0]
        return tf.constant(c, dtype=tf.float32, shape=var_shape)

def _apply_normalization(norm_type, x, *args):
    if norm_type == "batch_norm":
        pass

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
            input_channels = input_layer.get_shape().as_list[1]
        elif data_format == "NHWC":
            input_channels = input_layer.get_shape().as_list[3]
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
        W_shape = [input_layer.get_shape().as_list[1], num_outputs]
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