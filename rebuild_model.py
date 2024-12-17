import torch

def partial_reverse_tomodel(flattened, state_dict, train_layer):

    # 该部分是将展平后的参数重建为模型结构
    layer_idx = 0
    for name in state_dict:
        pa = state_dict[name]
        if name in train_layer:
            # print(f'Rebuild {name}')
            pa_shape = pa.shape
            pa_length = pa.view(-1).shape[0]
            pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
            layer_idx += pa_length
    return state_dict

def rebuild_model(input, model, train_layer):

    param = input
    target_num = 0
    state_dict = model.state_dict()

    for name in state_dict:
        module = state_dict[name]
        if name in train_layer:
            target_num += torch.numel(module)
    # print(target_num)

    params_num = torch.squeeze(param).shape[0]  # + 30720
    # print(params_num)
    assert (target_num == params_num) # 优先确定展平后的参数和net中待载入层的参数数量相等
    param = torch.squeeze(param)

    state_dict = partial_reverse_tomodel(param, state_dict, train_layer)
    model.load_state_dict(state_dict)

    return model