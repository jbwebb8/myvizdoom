import tensorflow as tf

def create_optimizer(opt_dict={}, **kwargs):
    """
    Arguments:
    - Opt_dict: dictionary of argument key/value pairs
        - type: type of optimizer
        - loss_list: list of loss functions to minimize
        - var_list: list of variables for which to compute/apply gradients
        - kwargs: dictionary of keyword arguments for optimizer
        - mods: dictionary of gradient modifications for layers
    - Optional args:
        - opt_type: type of optimizer
    """
    # Get optimizer type either through dict or kwarg
    try:
        opt_type = opt_dict["type"]
    except KeyError:
        opt_type = kwargs.pop("opt_type", None)
        if opt_type is None:
            raise SyntaxError("Optimizer type not provided.")
    
    # Avoid KeyError
    if "kwargs" not in opt_dict:
        opt_dict["kwargs"] = {}
    
    # Return optimizer of specific type
    if opt_type.lower() == "rmsprop":
        return rmsprop(**opt_dict["kwargs"], **kwargs)
    else:
        raise ValueError("Optimizer \"" + opt_type + "\" not yet defined.")

def rmsprop(learning_rate=0.001,
            decay=0.9,
            momentum=0.0,
            epsilon=1e-10):

    return tf.train.RMSPropOptimizer(learning_rate,
                                     decay=decay,
                                     momentum=momentum,
                                     epsilon=epsilon)

def create_train_step(ts_dict={}, **kwargs):
    # Get optimizer instance either through dict or kwarg
    try:
        opt = ts_dict["optimizer"]
    except KeyError:
        opt = kwargs.pop("optimizer", None)
        if opt is None:
            raise SyntaxError("Optimizer instance required.")
    
    # Avoid KeyError
    if "kwargs" not in ts_dict:
        ts_dict["kwargs"] = {}

    return _create_train_step(opt, **ts_dict["kwargs"], **kwargs)

def _modify_gradients(grad, mod_type, *params):
    if mod_type == "scale":
        return params[0] * grad
    elif mod_type == "clip_by_value":
        return tf.clip_by_value(grad, params[0], params[1])
    elif mod_type == "clip_by_norm":
        return tf.clip_by_norm(grad, params[0])
    else:
        raise ValueError("Gradient modification type \"" + mod_type
                            + "not yet defined.") 

def _check_list(arg):
    if isinstance(arg, list):
        return arg
    else:
        return [arg]

def _create_train_step(optimizer, 
                       loss_list=None,
                       loss_scope=None,
                       var_list=None,
                       var_scope=None,
                       mod_dict={}):
    # Minimize each loss function and backpropagate to variables their
    # gradients, which may be modified.
    root = tf.get_variable_scope().name
    if loss_list is None:
        loss_list = []
        for scope in _check_list(loss_scope):
            if scope is not None:
                scope = ('/').join([root, scope])
            losses = tf.get_collection(tf.GraphKeys.LOSSES, 
                                       scope=scope)
            if losses is None:
                print("Warning: no losses found for scope " + scope + " .")
            loss_list += losses
    if var_list is None:
        var_list = []
        for scope in _check_list(var_scope):
            if scope is not None:
                scope = ('/').join([root, scope])
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                    scope=scope)
            if var is None:
                print("Warning: no variables found for scope " + scope + " .")
            var_list += var

    train_step = []
    for l in loss_list:
        # Compute gradients for specified variables:
        # list of [grad(var), var]
        gvs = optimizer.compute_gradients(l, var_list=var_list)

        # Modify gradients for each layer in mod dictionary
        for layer_name, mod_list in mod_dict.items():
            clip_type = mod_list[0]
            params = mod_list[1:]
            with tf.name_scope("gradients/mod"):
                if layer_name.lower() == "all":
                    gvs = [[_modify_gradients(g, clip_type, *params), v]
                            for g, v in gvs]
                else:
                    gvs = [[_modify_gradients(g, clip_type, *params), v]
                            if layer_name in v.name else [g, v] for g, v in gvs]
        
        # Apply gradients after updating GraphKey class (used for batch norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step.append(optimizer.apply_gradients(gvs, name="train_step"))

    return train_step


    