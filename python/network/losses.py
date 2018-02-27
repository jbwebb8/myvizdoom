import tensorflow as tf

def create_loss_fn(input_loss=[], loss_dict={}, **kwargs):
    # Get loss type either through dict or kwarg
    try:
        loss_type = loss_dict["type"]
    except KeyError:
        loss_type = kwargs.pop("loss_type", None)
        if loss_type is None:
            raise SyntaxError("Loss type must be provided.")
    
    # Avoid KeyError
    if "kwargs" not in loss_dict:
        loss_dict["kwargs"] = {}

    # Return loss of specific type
    if loss_type.lower() == "mean_squared_error":
        return mean_squared_error(*input_loss, **loss_dict["kwargs"], **kwargs)
    elif loss_type.lower() == "huber":
        return huber_loss(*input_loss, **loss_dict["kwargs"], **kwargs)
    elif loss_type.lower() == "advantage":
        return advantage_loss(*input_loss, **loss_dict["kwargs"], **kwargs)
    elif loss_type.lower() == "softmax_cross_entropy_with_logits":
        return softmax_cross_entropy_with_logits(*input_loss, **loss_dict["kwargs"], **kwargs)
    else:
        raise ValueError("Loss function \"" + loss_type + "\" not yet defined.")

def _check_list(arg):
    if isinstance(arg, list):
        try:
            return arg[0], arg[1:]
        except IndexError:
            return arg[0], []
    else:
        return arg, []

def _get_weighted_error(target, prediction, weights, name="error"):
    error = tf.subtract(target, prediction, name=name)
    if weights is not None:
        error = weights * error
    return error

def mean_squared_error(target,
                       prediction,
                       weights=None,
                       add_to_collection=True,
                       scope="loss"):
    with tf.name_scope(scope):
        error = _get_weighted_error(target, prediction, weights)
        mse = tf.reduce_mean(tf.square(error), name="mse")
        if add_to_collection:
            tf.add_to_collection(tf.GraphKeys.LOSSES, mse)
        return mse

def huber_loss(target,
               prediction,
               delta=1.0,
               weights=None,
               add_to_collection=True,
               scope="loss"):
    with tf.name_scope(scope):
        error = _get_weighted_error(target, prediction, weights)
        huber_loss = tf.where(tf.abs(error) < delta, 
                              0.5*tf.square(error),
                              delta*(tf.abs(error) - 0.5*delta),
                              name="huber_loss")
        if add_to_collection:
            tf.add_to_collection(tf.GraphKeys.LOSSES, huber_loss)
        return huber_loss

def softmax_cross_entropy_with_logits(target,
                                      logits,
                                      sum_axis=-1,
                                      weights=None,
                                      add_to_collection=True,
                                      scope="loss"):
    with tf.name_scope(scope):
        logits_shifted = logits - tf.reduce_max(logits, axis=sum_axis, keep_dims=True)
        sum_exp = tf.reduce_sum(tf.exp(logits_shifted)) # Σe^x
        log_sum_exp = tf.log(sum_exp, axis=sum_axis, keep_dims=True) # ln(Σe^x)
        xent = -tf.reduce_sum(tf.multiply(target, (logits_shifted - log_sum_exp)))
        if add_to_collection:
            tf.add_to_collection(tf.GraphKeys.LOSSES, xent)
        return weights * xent

### I think this can be broken into entropy loss and cross entropy loss?
def advantage_loss(q,
                   v,
                   pi,
                   actions,
                   beta=0.1,
                   weights=None,
                   add_to_collection=True,
                   scope="loss"):
    with tf.name_scope(scope):
        pi_a = tf.gather_nd(pi, actions, name="pi_a")

        # Note the negative sign on adv_loss and positive (actually double
        # negative) on entropy_loss to use grad descent instead of ascent
        adv_loss = -tf.multiply(tf.log(pi_a), error, name="advantage_loss")
        entropy_loss = tf.reduce_sum(pi * tf.log(pi), name="entropy_loss")
        pi_loss_fn = tf.add(adv_loss, beta * entropy_loss,
                            name="policy_loss")
        if add_to_collection:
            tf.add_to_collection(tf.GraphKeys.LOSSES, pi_loss_fn)
        return weights * pi_loss_fn