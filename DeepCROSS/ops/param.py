import numpy as np
import tensorflow as tf

import locale
locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}

def param(name, *args, **kwargs):

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    result = _params[name]
    i = 0
    while result in _param_aliases:
        i += 1
        result = _param_aliases[result]
    return result

def params_with_name(name):
    return [p for n,p in _params.items() if name in n]

def delete_all_params():
    _params.clear()

def alias_params(replace_dict):
    for old,new in replace_dict.items():
        _param_aliases[old] = new

def delete_param_aliases():
    _param_aliases.clear()
