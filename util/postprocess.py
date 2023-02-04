import torch.nn.functional as F


def backbone_output_activation(params, data):
    x = data['offset_pred']

    if not params.is_output_activation:
        return x

    mpp = params.max_shift_pixels
    act = params.output_activation
    if act == 'tanh':
        x = x.tanh() * mpp
    elif act == 'hardtanh':
        x = F.hardtanh(x, -mpp, mpp)
    elif act == 'sigmoid':
        x = (F.sigmoid(x) * 2 - x.new_ones(*x.shape)) * mpp
    elif act == 'hardsigmoid':
        x = (F.hardsigmoid(x) * 2 - x.new_ones(*x.shape)) * mpp
    else:
        raise

    data['offset_pred'] = x

    return data
