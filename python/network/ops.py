import tensorflow as tf

# Ops needed:
# - tf.gather_nd for advantage loss, best action, etc.

def create_op(input_op=[], op_dict={}, **kwargs):
    # Get operation type either through dict or kwarg
    try:
        op_type = op_dict["type"]
    except KeyError:
        op_type = kwargs.pop("op_type", None)
        if op_type is None:
            raise SyntaxError("Operation type must be provided.")
    
    # Avoid KeyError
    if "kwargs" not in op_dict:
        op_dict["kwargs"] = {}

    # Return operation of specific type
    if op_type.lower() == "gather_nd":
        return gather_nd(*input_op, **op_dict["kwargs"], **kwargs)
    elif op_type.lower() == "rnn_loss_mask":
        return rnn_loss_mask(*input_op, **op_dict["kwargs"], **kwargs)
    else:
        raise ValueError("Operation type \"" + op_type + "\" not yet defined.")

def gather_nd(params,
              indices,
              name="gather_nd"):
    return tf.gather_nd(params, indices, name=name)

def rnn_loss_mask(weights,
                  mask_len,
                  batch_size=None,
                  scope="loss_mask"):
    with tf.name_scope(scope):
        # Get batch size and trace length of RNN output
        ndim = tf.rank(weights)
        if ndim < 2: # tf.shape(weights) = [batch_size*tr_len]
            if batch_size is None:
                raise ValueError("batch_size required if weights flattened.")
            tr_len = tf.shape(weights)[0] // batch_size
        elif ndim == 2: # tf.shape(weights) = [batch_size, tr_len]
            tr_len = tf.shape(weights)[1]
        else:
            raise ValueError("Rank of weights tensor must be <= 2.")

        # Create mask to block first mask_len values
        mask_len = tf.minimum(mask_len, tr_len)
        mask_zeros = tf.zeros([batch_size, mask_len])
        mask_ones = tf.ones([batch_size, tr_len - mask_len])
        mask = tf.concat([mask_zeros, mask_ones], axis=1)
        mask = tf.reshape(mask, tf.shape(weights))
        
        # Apply mask to weights
        masked_weights = mask * weights
        
        return masked_weights