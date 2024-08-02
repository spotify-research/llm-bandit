
def vector_to_param_dict(vec, named_parameters_orig):
    pointer = 0
    param_dict = {}
    for name, param in named_parameters_orig:
        num_param = param.numel()
        param_dict[name] = vec[pointer:pointer + num_param].view_as(param)
        pointer += num_param

    return param_dict


def vector_to_param_dict_last_layer(vec, layer_name, layer_shape):
    param_dict = {layer_name: vec.reshape(layer_shape)}

    return param_dict


