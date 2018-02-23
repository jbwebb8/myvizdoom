def recursive_dict_search(dictionary, old_value, new_value):
    """
    Replaces all key-old_value pairs with key-new_value pairs in dictionary.
    
    Args:
    - d: Dictionary to be modified.
    - key: Initial key.
    - old_value: Value to be replaced.
    - new_value: Value to replace old_value.

    Returns:
    - None (dictionary modified in place)
    """
    _recursive_dict_search(dictionary, dictionary, [], old_value, new_value)

def _recursive_dict_search(init_d, d, keys, old_val, new_val):
    # If dictionary, continue to explore all key-value pairs
    if isinstance(d, dict):
        for k, v in zip(d.keys(), d.values()):
            keys.append(k)
            _recursive_dict_search(init_d, v, keys, old_val, new_val)
        if len(keys) > 0: # avoids error at end
            keys.pop()
    
    # If list, then value list iterated through for all items
    elif isinstance(d, list):
        d = _recursive_list_search(d, old_val, new_val)
        t = init_d 
        for key in keys[:-1]:
            t = t[key]
        t[keys[-1]] = d
        keys.pop()
    
    # Otherwise, then replace old value with new value
    else:
        if d == old_val:
            t = init_d 
            for key in keys[:-1]:
                t = t[key]
            t[keys[-1]] = new_val
        keys.pop()

def _recursive_list_search(l, old_val, new_val):
    if isinstance(l, list):
        t = []
        for l_ in l:
            v = _recursive_list_search(l_, old_val, new_val)
            t.append(v)
        return t
    elif isinstance(l, dict):
        _recursive_dict_search(l, l, [], old_val, new_val)
        return l
    else:
        if l == old_val:
            l = new_val
        return l
