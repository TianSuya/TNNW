import torch

def flatten_model(model, train_layer):
    params = []
    state_dict = model.state_dict()
    for name in state_dict:
        param = state_dict[name]
        if name in train_layer:
            params.append(param.reshape(-1))
    flattened_params = torch.cat(params, 0)
    return flattened_params